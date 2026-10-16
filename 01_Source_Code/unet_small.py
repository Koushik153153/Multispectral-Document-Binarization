import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional but recommended; required for pretrained ResNet50.
from torchvision.models import resnet50, ResNet50_Weights


class BandReweight(nn.Module):
    def __init__(self, bands: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, bands // reduction)
        self.fc1 = nn.Linear(bands, hidden)
        self.fc2 = nn.Linear(hidden, bands)

    def forward(self, x, return_weights: bool = False):
        s = x.mean(dim=(2, 3))  # (B,C)
        w = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))  # (B,C)
        out = x * w.unsqueeze(-1).unsqueeze(-1)
        if return_weights:
            return out, w
        return out


class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """
    Upsample -> concat skip -> conv -> conv
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvRelu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))
        return x


def _adapt_first_conv_weights(conv_w: torch.Tensor, in_ch: int) -> torch.Tensor:
    """
    conv_w: (out_ch=64, in_ch=3, k, k) from pretrained ResNet
    Return: (64, in_ch, k, k) by repeating RGB weights over channels.
    This matches Hollaus 2019’s described strategy of repeating weights for multispectral input. [file:594]
    """
    if in_ch == 3:
        return conv_w

    # Repeat along channel dim and trim
    repeat = (in_ch + 2) // 3
    w = conv_w.repeat(1, repeat, 1, 1)[:, :in_ch, :, :].contiguous()

    # Keep activation scale similar
    w = w * (3.0 / float(in_ch))
    return w


class UNetSmall(nn.Module):
    """
    ResNet50 encoder + U-Net decoder for 1-channel output (FG1 mask).
    Keeps the same public API as your previous UNetSmall.

    Args:
      in_ch: 12 for MSBIN, or 1 if --white_only
      band_reweight: apply simple squeeze-excitation style band weights before encoder
      band_drop_p: Dropout2d on input channels (channel dropout)
      pretrained: use ImageNet pretrained ResNet50 weights
      freeze_encoder: optionally freeze encoder for speed (usually False for best FM)
    """
    def __init__(
        self,
        in_ch=12,
        base=16,                 # kept for compatibility; not used (ResNet defines widths)
        band_reweight=False,
        band_drop_p=0.0,
        pretrained=True,
        freeze_encoder=False,
    ):
        super().__init__()

        self.band_reweight = BandReweight(in_ch) if band_reweight else nn.Identity()
        self.band_drop = nn.Dropout2d(p=band_drop_p) if band_drop_p > 0 else nn.Identity()
        self.last_band_weights = None
        self.freeze_encoder = bool(freeze_encoder)

        # ---- Encoder: ResNet50 ----
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        enc = resnet50(weights=weights)  # torchvision API [web:687]

        # Replace first conv to accept in_ch channels (12 for MSBIN)
        old_conv1 = enc.conv1
        enc.conv1 = nn.Conv2d(in_ch, old_conv1.out_channels,
                              kernel_size=old_conv1.kernel_size,
                              stride=old_conv1.stride,
                              padding=old_conv1.padding,
                              bias=False)

        if pretrained:
            with torch.no_grad():
                enc.conv1.weight.copy_(_adapt_first_conv_weights(old_conv1.weight, in_ch))

        self.enc = enc

        # ResNet “stages” for skip connections
        # x0: after conv1+bn+relu  -> 64, H/2
        # x1: after maxpool+layer1 -> 256, H/4
        # x2: after layer2         -> 512, H/8
        # x3: after layer3         -> 1024, H/16
        # x4: after layer4         -> 2048, H/32
        self.encoder_out_channels = (64, 256, 512, 1024, 2048)

        if freeze_encoder:
            for p in self.enc.parameters():
                p.requires_grad = False
            # keep BN stable in frozen encoder
            self.enc.eval()

        # ---- Decoder ----
        # decoder blocks: (x4 -> x3 -> x2 -> x1 -> x0)
        self.dec4 = DecoderBlock(in_ch=2048, skip_ch=1024, out_ch=512)
        self.dec3 = DecoderBlock(in_ch=512,  skip_ch=512,  out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256,  skip_ch=256,  out_ch=128)
        self.dec1 = DecoderBlock(in_ch=128,  skip_ch=64,   out_ch=64)

        # final upsample to original resolution (H, W)
        self.head_conv1 = ConvRelu(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.enc.eval()
        return self

    def forward(self, x):
        # Band reweight logging (same behavior as your old model)
        if hasattr(self.band_reweight, "forward") and not isinstance(self.band_reweight, nn.Identity):
            x, w = self.band_reweight(x, return_weights=True)
            self.last_band_weights = w.detach()
        else:
            x = self.band_reweight(x)
            self.last_band_weights = None

        x = self.band_drop(x)

        # --- ResNet forward with feature taps ---
        x0 = self.enc.relu(self.enc.bn1(self.enc.conv1(x)))   # 64, H/2
        x1 = self.enc.layer1(self.enc.maxpool(x0))            # 256, H/4
        x2 = self.enc.layer2(x1)                               # 512, H/8
        x3 = self.enc.layer3(x2)                               # 1024, H/16
        x4 = self.enc.layer4(x3)                               # 2048, H/32

        # --- Decoder with skips ---
        d4 = self.dec4(x4, x3)  # 512, H/16
        d3 = self.dec3(d4, x2)  # 256, H/8
        d2 = self.dec2(d3, x1)  # 128, H/4
        d1 = self.dec1(d2, x0)  # 64,  H/2

        # upsample to input size
        d0 = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.head_conv1(d0)
        return self.out(d0)
