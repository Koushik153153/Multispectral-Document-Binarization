"""Sequential controlled GPU experiments; explicit run/evaluate/compare operations."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys

from pixel_metrics import PixelCounts
from training_support import atomic_write, write_json, read_checkpoint

VARIANTS = {"baseline": (False, 0), "reweight_only": (True, 0),
            "drop_only": (False, .2), "reweight_and_drop": (True, .2)}
COMMON = {"fg_type", "patch", "stride", "batch", "num_workers", "cache_mb", "precision", "threads",
          "min_fg_frac", "lr", "pos_weight", "bce_w", "dice_w", "grad_clip", "val_thr"}
ROOT = Path(__file__).resolve().parent


def load_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("format_version") != 2 or set(spec.get("common", {})) != COMMON:
        raise ValueError("Unsupported/incomplete experiment specification")
    if type(spec["epochs"]) is not int or spec["epochs"] < 1:
        raise ValueError("Epoch budget must be positive")
    if not spec["seeds"] or any(type(s) is not int or not 0 <= s < 2**32 - 1000003 for s in spec["seeds"]) or len(set(spec["seeds"])) != len(spec["seeds"]):
        raise ValueError("Seeds must be unique nonnegative integers")
    if set(spec["variants"]) != set(VARIANTS):
        raise ValueError("Expected exactly four controlled variants")
    for name, (attention, drop) in VARIANTS.items():
        if spec["variants"][name] != {"band_reweight": attention, "band_drop_p": drop}:
            raise ValueError(f"Invalid controlled variant: {name}")
    fixed = {"fg_type": 1, "patch": 256, "stride": 256, "min_fg_frac": 0, "val_thr": .5}
    if any(spec["common"][k] != v for k, v in fixed.items()):
        raise ValueError("Task/patch/filter/threshold differs from controlled FG1 protocol")
    return spec


def training_command(spec, name, seed, data, splits, output, resume=False):
    if name not in VARIANTS or seed not in spec["seeds"]:
        raise ValueError("Unknown configuration or seed")
    command = [sys.executable, "-u", str(ROOT / "train_unet_msbin_cpu.py"),
               "--msbin_root", str(Path(data).resolve()), "--train_keys", str(Path(splits).resolve() / "train_keys.txt"),
               "--val_keys", str(Path(splits).resolve() / "val_keys.txt"), "--outdir", str(output),
               "--epochs", str(spec["epochs"]), "--seed", str(seed), "--experiment_name", name,
               "--pretrained", "--require_cuda"]
    for key, value in spec["common"].items():
        command.extend(["--" + key, str(value)])
    variant = spec["variants"][name]
    command.extend(["--band_drop_p", str(variant["band_drop_p"])])
    if variant["band_reweight"]:
        command.append("--band_reweight")
    if resume:
        command.append("--resume")
    return command


def checked_scores(scores):
    if any(type(scores.get(k)) is not int or scores[k] < 0 for k in ("tp", "fp", "fn")):
        raise ValueError("Invalid pixel counts")
    actual = PixelCounts(*(scores[k] for k in ("tp", "fp", "fn"))).scores()
    if any(not math.isclose(scores[k], actual[k], abs_tol=1e-12) for k in ("precision", "recall", "f1")):
        raise ValueError("Metrics do not match pixel counts")
    return actual


def validate_run_selection(saved, spec, name, seed):
    """Do not silently compare a different run copied under the requested folder."""
    if (saved.get("experiment_name") != name or saved.get("seed") != seed or
            saved.get("epochs") != spec["epochs"] or
            any(saved.get(k) != v for k, v in spec["common"].items())):
        raise ValueError("Run arguments differ from selected experiment specification/configuration/seed")


def comparison_rows(manifests, expected_pages=20):
    rows, per_page, reference, init_by_seed, seen = [], [], None, {}, set()
    for manifest in manifests:
        if manifest.get("status") != "complete" or manifest.get("format_version") != 2:
            raise ValueError("Comparison requires completed version 2 evaluations; historical results are separate")
        args = manifest.get("comparison_training_args", manifest["training_args"])
        name, seed = args["experiment_name"], args["seed"]
        if name not in VARIANTS or (name, seed) in seen:
            raise ValueError("Unknown/duplicate configuration and seed")
        seen.add((name, seed))
        if (manifest["model_config"]["band_reweight"], manifest["model_config"]["band_drop_p"]) != VARIANTS[name]:
            raise ValueError("Configuration label and model variant differ")
        if (manifest["model_config"].get("in_ch") != 12 or
                manifest["model_config"].get("pretrained") is not True or
                manifest["model_config"].get("freeze_encoder") is not False):
            raise ValueError("Comparison requires the pretrained, unfrozen, 12-band protocol")
        keys = manifest["splits"]["val"]["keys"]
        if manifest["keys_file_sha256"] != manifest["splits"]["val"]["sha256"]:
            raise ValueError("Evaluation keys differ from the checkpoint's exact validation split bytes")
        page_keys = [p["key"] for p in manifest["pages"]]
        if (len(keys) != expected_pages or len(set(keys)) != expected_pages or
                len(page_keys) != expected_pages or len(set(page_keys)) != expected_pages or
                set(page_keys) != set(keys) or len(manifest["keys"]) != expected_pages or set(manifest["keys"]) != set(keys)):
            raise ValueError("Comparison requires all validation pages exactly once")
        if manifest["source_split"] != "train" or manifest["fg_type"] != 1 or manifest["threshold"] != .5 or manifest["tiling"]["tile"] != 256 or manifest["tiling"]["stride"] != 256:
            raise ValueError("Evaluation task/settings differ from fixed protocol")
        if manifest["precision"] not in ("fp32", "amp") or manifest["precision"] != args["precision"]:
            raise ValueError("Missing/inconsistent precision provenance")
        ignored = {"outdir", "msbin_root", "train_keys", "val_keys", "resume", "seed", "experiment_name", "band_reweight", "band_drop_p"}
        common_args = {k: v for k, v in args.items() if k not in ignored}
        model_common = {k: v for k, v in manifest["model_config"].items() if k not in ("band_reweight", "band_drop_p")}
        protocol = dict(splits=manifest["splits"], task=manifest["fg_type"], tiling=manifest["tiling"],
                        threshold=manifest["threshold"], rule=manifest["threshold_rule"],
                        conventions=manifest["conventions"], source_split=manifest["source_split"],
                        evaluation_source=manifest["evaluation_source"], device=manifest["device"],
                        evaluation_versions=manifest["evaluation_versions"], evaluation_backend=manifest["evaluation_backend"],
                        model=model_common, arguments=common_args, source=manifest["source_identity"], versions=manifest["versions"])
        if reference is not None and protocol != reference:
            raise ValueError("Mismatched splits, task, training configuration, source, or evaluation settings")
        reference = protocol
        identity = manifest["initialization"]["sha256"]
        if seed in init_by_seed and init_by_seed[seed] != identity:
            raise ValueError("Shared initialization differs within seed")
        init_by_seed[seed] = identity
        pooled = PixelCounts()
        values = []
        for page in manifest["pages"]:
            score = checked_scores(page["scores"])
            for key in ("tp", "fp", "fn"):
                setattr(pooled, key, getattr(pooled, key) + score[key])
            values.append(score)
            per_page.append(dict(configuration=name, seed=seed, page=page["key"],
                                 selected_epoch=manifest["completed_epoch"],
                                 checkpoint_sha256=manifest["checkpoint"]["sha256"], **score))
        aggregate = checked_scores(manifest["aggregate_scores"])
        if aggregate != pooled.scores():
            raise ValueError("Pooled counts do not match per-page counts")
        if not 1 <= manifest["completed_epoch"] <= args["epochs"]:
            raise ValueError("Selected epoch outside budget")
        if manifest.get("run_completed_epoch") != args["epochs"]:
            raise ValueError("Comparison requires the completed common epoch budget")
        rows.append(dict(configuration=name, seed=seed, selected_epoch=manifest["completed_epoch"],
                         precision_mode=manifest["precision"], checkpoint_sha256=manifest["checkpoint"]["sha256"],
                         checkpoint_path=manifest["checkpoint"]["path"], shared_initialization_sha256=identity,
                         **aggregate, **{"page_mean_" + k: sum(v[k] for v in values) / len(values) for k in ("precision", "recall", "f1")}))
    if not rows:
        raise ValueError("No evaluation manifests supplied")
    return dict(scope="pooled valid pixels on all validation pages; page_mean columns are unweighted page means; not test results",
                runs=rows, pages=per_page, protocol=reference)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "evaluate", "compare"))
    p.add_argument("--spec", type=Path, default=ROOT / "experiments.json")
    p.add_argument("--configuration", choices=(*VARIANTS, "all"), default="baseline")
    p.add_argument("--data", type=Path, default=ROOT / "MSBin")
    p.add_argument("--splits", type=Path, default=ROOT / "splits")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    spec = load_spec(args.spec)
    names = list(VARIANTS) if args.configuration == "all" else [args.configuration]
    if args.resume and (args.action != "run" or len(names) * len(spec["seeds"]) != 1):
        p.error("Resume exactly one configuration/seed explicitly (use a one-seed spec)")
    if args.action == "run":
        from training_support import check_output_dir
        for seed in spec["seeds"]:
            for name in names:
                check_output_dir(args.runs.resolve() / f"{name}_seed{seed}", args.resume)
    manifests = []
    for seed in spec["seeds"]:
        for name in names:
            folder = args.runs.resolve() / f"{name}_seed{seed}"
            if args.action == "run":
                check_output_dir(folder, args.resume)
                command = training_command(spec, name, seed, args.data, args.splits, folder, args.resume)
                print("Launching sequentially:", command, flush=True)
                subprocess.run(command, check=True, cwd=ROOT)
            elif args.action == "evaluate":
                last = read_checkpoint(folder / "last.pt")
                if last["format_version"] != 2 or last["completed_epoch"] != spec["epochs"] or last.get("software_smoke_only"):
                    raise ValueError("Evaluate only a completed version 2 research run at the selected budget")
                saved = last["training_args"]
                validate_run_selection(saved, spec, name, seed)
                selected_epoch = max(last["history"], key=lambda row: row["patch_val_f1"])["epoch"]
                completed = last["completed_epoch"]
                provenance = (last["splits"], last["source_identity"], last["initialization"])
                del last
                subprocess.run([sys.executable, "-u", str(ROOT / "infer_msbin_cpu.py"),
                                "--msbin_root", str(args.data.resolve()), "--keys_txt", str(args.splits.resolve() / "val_keys.txt"),
                                "--ckpt", str(folder / "best.pt"), "--outdir", str(folder / "validation_all"),
                                "--tile", "256", "--stride", "256", "--thr", "0.5", "--device", "cuda", "--compare_gt"], check=True, cwd=ROOT)
                manifest_path = folder / "validation_all/manifest.json"
                evaluated = json.loads(manifest_path.read_text(encoding="utf-8"))
                operational = {"epochs", "resume", "outdir", "train_keys", "val_keys"}
                best_args = {k: v for k, v in evaluated["training_args"].items() if k not in operational}
                last_args = {k: v for k, v in saved.items() if k not in operational}
                if (evaluated["completed_epoch"] != selected_epoch or best_args != last_args or
                        (evaluated["splits"], evaluated["source_identity"], evaluated["initialization"]) != provenance):
                    raise ValueError("best.pt does not match last.pt's validation-selected epoch/configuration")
                evaluated["run_completed_epoch"] = completed
                evaluated["comparison_training_args"] = saved
                write_json(manifest_path, evaluated)
            else:
                manifest = json.loads((folder / "validation_all/manifest.json").read_text(encoding="utf-8"))
                validate_run_selection(manifest.get("comparison_training_args", manifest["training_args"]), spec, name, seed)
                manifests.append(manifest)
    if args.action == "compare":
        if args.output is None:
            p.error("--output required for comparison (new directory)")
        report = comparison_rows(manifests)
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "comparison.json", report)
        for name, rows in (("comparison.csv", report["runs"]), ("per_page.csv", report["pages"])):
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            atomic_write(args.output / name, lambda out: out.write(stream.getvalue().encode()))


if __name__ == "__main__":
    main()
