# Multispectral Document Binarization Using Band Reweighting and Band Dropping

A deep learning project for pixel-level binarization of multispectral document images using a ResNet50 encoder and U-Net decoder. The project investigates whether learning the relative importance of spectral bands and randomly dropping bands during training can improve document foreground extraction.

## 1. Project Overview

Document binarization is the process of separating foreground text from the background of a document image. It is an important preprocessing step for digitizing historical, faded, or degraded documents.

Traditional document binarization methods commonly operate on grayscale or RGB images. However, different spectral bands of a document may reveal differences between ink, paper, stains, and other visual characteristics.

This project uses the **MSBin multispectral document dataset**, in which each document page is represented by 12 aligned grayscale spectral bands.

A deep learning segmentation model processes these 12 bands and predicts a binary mask that identifies foreground text pixels.

The project compares four experimental configurations:

1. Baseline model
2. Band Reweighting
3. Band Dropping
4. Band Reweighting + Band Dropping

The purpose is to evaluate how these strategies affect the precision, recall, and F1-score of document binarization.

## 2. Problem Statement

A single grayscale image may not contain all the useful visual information available across multiple spectral bands. Some bands may provide more useful information about the separation between text and background than others.

The project investigates two strategies:

- **Band Reweighting:** Allows the model to learn the relative importance of the input spectral bands.
- **Band Dropping:** Randomly suppresses selected spectral channels during training to encourage the model to avoid depending too heavily on a fixed subset of bands.

The effectiveness of these strategies is evaluated by comparing them against a baseline model under a controlled experimental setup.

## 3. Project Objectives

- Process multispectral document pages containing 12 aligned grayscale bands.
- Develop a deep learning model for pixel-level foreground segmentation.
- Implement and compare Band Reweighting and Band Dropping.
- Train and validate four model configurations.
- Prevent data leakage by separating training, validation, and test data at the document-page level.
- Evaluate predictions using Precision, Recall, and F1-score.
- Generate full-page binary masks and visual comparisons.
- Analyze whether the proposed strategies improve binarization performance.

## 4. Dataset: MSBin

**Dataset name:** MSBin multispectral document image dataset

**Dataset DOI:** [10.5281/zenodo.3257366](https://doi.org/10.5281/zenodo.3257366)

The dataset contains multispectral document images represented through 12 aligned grayscale spectral bands per page.

### What is a spectral band?

A spectral band is an image channel that records the document at a particular spectral range. Each band represents the same document page, but its visual information may differ from that of other bands.

For example, faded ink or marks that are difficult to distinguish in one band may be more distinguishable in another. The model learns from the combined spectral information rather than relying on a single grayscale image.

### Dataset labels

The project uses ground-truth annotations to determine which pixels should be classified as foreground or background.

- **FG1:** Target foreground class used for binarization.
- **FG2:** Treated as valid background for the FG1 task.
- **UR:** Uncertain region. These pixels are excluded from pixel-level metric calculations.

The model predicts the FG1 foreground mask. It does not perform text transcription or optical character recognition (OCR).

### Dataset split

The project uses page-level separation to prevent patches from the same document page from appearing across different splits.

| Split | Number of pages | Purpose |
|---|---:|---|
| Training | 60 | Learn model parameters |
| Validation | 20 | Monitor performance and select checkpoints |
| Held-out testing | 50 | Final performance evaluation |
| **Total** | **130** | **Separate page-level splits** |

The dataset itself is not included in this GitHub repository. Download it from the official dataset source and prepare the expected folder structure before attempting to reproduce the experiments.

## 5. Model Architecture

The project uses a **ResNet50 encoder with a U-Net decoder** for image segmentation.

### ResNet50 encoder

ResNet50 is a convolutional neural network used to extract visual features from the input image. It learns useful patterns such as edges, textures, and higher-level spatial features.

In this project, the encoder processes the multispectral input.

### U-Net decoder

U-Net is an encoder-decoder architecture designed for pixel-level image segmentation. The decoder reconstructs spatial information and generates a foreground probability map. Skip connections help preserve useful spatial details from earlier stages of the network.

### Model output

The model produces a foreground probability for each pixel. A fixed threshold of 0.5 is used to convert these probabilities into a binary mask:

- Probability at or above 0.5: foreground.
- Probability below 0.5: background.

The resulting mask identifies the predicted foreground regions of the document.

## 6. Experimental Configurations

Four configurations are compared to understand the contribution of each strategy.

### Experiment 1: Baseline

The baseline uses the ResNet50-U-Net segmentation architecture without the additional Band Reweighting or Band Dropping strategies.

It provides the reference performance for the other experiments.

### Experiment 2: Band Reweighting Only

This configuration introduces learnable weights for the spectral bands. The model can adjust their relative contributions instead of treating every band as equally important.

The purpose is to investigate whether adaptive spectral weighting improves foreground extraction.

### Experiment 3: Band Dropping Only

This configuration randomly suppresses selected spectral channels during training.

The goal is to encourage the model to learn useful representations from different combinations of bands rather than depending too heavily on a fixed subset.

Band Dropping is a training-time strategy. During normal evaluation and inference, all 12 input bands are provided.

### Experiment 4: Band Reweighting + Band Dropping

This configuration combines both strategies. It investigates whether learning spectral importance and suppressing selected bands during training provide complementary benefits.

### Experimental controls

The experiments use the same general segmentation task, page-level data split, 256 × 256 training patches, and evaluation methodology. Each configuration was trained for 10 epochs using seed 42.

The best checkpoint for each configuration was selected using validation performance.

Only one training seed was used for each configuration. Therefore, the results describe the tested runs and do not establish statistical significance.

## 7. Training Methodology

The training pipeline follows these general stages:

1. Load the multispectral document pages and their corresponding ground-truth masks.
2. Prepare and normalize the spectral input.
3. Construct 256 × 256 patches for training.
4. Feed the 12-channel input into the segmentation network.
5. Generate pixel-level foreground predictions.
6. Calculate the training loss using weighted Binary Cross-Entropy (BCE) and Dice loss.
7. Update the model parameters using the AdamW optimizer.
8. Evaluate the model on the validation split.
9. Select and save the best checkpoint for each configuration.

### Loss functions

**Weighted Binary Cross-Entropy (BCE):** Measures the difference between the predicted foreground probabilities and the ground-truth labels while applying class weights.

**Dice loss:** Encourages overlap between the predicted foreground region and the ground-truth foreground region.

Using both losses helps the model learn pixel-level classification while considering foreground-region overlap.

### Optimizer

AdamW updates the model parameters during training and uses decoupled weight decay to help regularize the model.

## 8. Evaluation Metrics

The models are evaluated using Precision, Recall, and F1-score on valid pixels. Pixels marked as uncertain regions are excluded from these metric calculations.

### Precision

Precision measures how many pixels predicted as foreground are actually foreground.

Higher precision means fewer background pixels are incorrectly classified as foreground.

### Recall

Recall measures how many actual foreground pixels are correctly detected by the model.

Higher recall means fewer foreground pixels are missed.

### F1-score

F1-score is the harmonic mean of Precision and Recall. It provides a combined measure of these two metrics.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

The F1-score is reported as a value between 0 and 1, where a higher value indicates a better balance between Precision and Recall.

**Important:** F1-score is not the same as pixel accuracy.

## 9. Official Held-Out Test Results

The following results are from the project's official evaluation on the 50 held-out test pages.

| Model configuration | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Baseline | 0.81115 | 0.94617 | 0.87347 |
| Band Reweighting Only | 0.83267 | 0.94022 | **0.88318** |
| Band Dropping Only | 0.82433 | 0.93480 | 0.87610 |
| Band Reweighting + Band Dropping | 0.81805 | 0.94165 | 0.87551 |

### Interpretation

- **Baseline:** Achieved an F1-score of 0.87347 and provides the reference result.
- **Band Reweighting Only:** Achieved the highest held-out test F1-score of 0.88318 among the four tested configurations.
- **Band Dropping Only:** Achieved an F1-score of 0.87610, slightly above the baseline.
- **Combined Strategy:** Achieved an F1-score of 0.87551, above the baseline but below the Band Reweighting-only result.

### Main finding

In the reported experiments, Band Reweighting alone produced the strongest overall Precision-Recall balance according to F1-score.

Combining Band Reweighting and Band Dropping did not outperform Band Reweighting alone in this particular run.

These findings should be interpreted carefully because each configuration was trained using only one seed. Multiple independent runs would be needed to determine whether the observed differences are consistent.

The held-out test set is intended for final evaluation, not repeated model tuning.

## 10. Repository Structure

The repository is organized into folders according to their role in the project.

```text
Multispectral-Document-Binarization/
│
├── .gitignore
│
├── 01_Source_Code/
│   ├── Model architecture
│   ├── Training and experiment scripts
│   ├── Inference utilities
│   └── Evaluation and supporting utilities
│
├── 04_Validation_Results/
│   └── Validation outputs and experiment results
│
├── 05_Test_Results/
│   └── Held-out test evaluation outputs
│
├── 06_Experiment_Logs/
│   └── Experiment records and logs
│
├── 07_Documentation/
│   └── Supporting project documentation
│
└── demo_test/
    ├── run_demo.py
    └── Demo-related scripts and supporting files
```

The descriptions above explain each folder's purpose. The exact files inside the folders should be inspected in the repository for their individual implementation details.

### `01_Source_Code`

Contains the implementation of the project, including the model, training and inference code, and supporting utilities.

Notable scripts include:

- `unet_small.py`: Model architecture implementation.
- `infer_msbin_cpu.py`: CPU-oriented inference utility.
- `demo_test/run_demo.py`: Runs the demonstration pipeline using the trained model checkpoints.

The source code contains the implementation logic used to process the multispectral input and generate segmentation predictions.

### `04_Validation_Results`

Contains validation-related outputs used to examine model performance and support checkpoint selection.

Validation results help determine which checkpoint to retain for each experimental configuration.

### `05_Test_Results`

Contains test-related evaluation outputs for examining the models' final performance.

These results support the comparison of the four configurations using the chosen evaluation metrics.

### `06_Experiment_Logs`

Contains experiment records and logs that help document how the experiments were run and what outputs were produced.

### `07_Documentation`

Contains supporting documentation related to the project and its implementation.

### `demo_test`

Contains scripts for running a demonstration of the trained models. The demonstration can generate predicted masks, comparisons, and metric summaries for the selected sample pages when the required data and checkpoints are available.

Generated demo output directories are excluded from Git to avoid adding further large files.

### `.gitignore`

Specifies files and folders that should not be included in Git, including the dataset, presentation/report folder, Python cache files, generated demo outputs, and large model checkpoints.

## 11. Demonstration Pipeline

The project includes a demonstration script that runs inference using the four trained configurations on selected document pages.

The recorded demonstration uses pages BT11 and BT15.

The demonstration pipeline:

1. Loads the selected document page and its 12 spectral bands.
2. Loads the trained model checkpoint for each configuration.
3. Runs tiled, full-page inference.
4. Converts predicted probabilities into binary masks using a threshold of 0.5.
5. Generates comparison outputs and metric summaries.
6. Records the results for the selected pages.

The recorded demonstration was run on CPU. Inference time therefore depends on the hardware, image dimensions, and inference settings.

**Demonstration results are separate from official held-out test results.** The demonstration uses only two selected pages and should not be used as a replacement for the evaluation on all 50 held-out test pages.

## 12. Dataset and Model Checkpoint Availability

To keep the GitHub repository manageable, the following items were excluded from the upload:

- `02_Dataset_and_Splits/`: Dataset files and split data.
- `08_Presentation_and_Report/`: Presentation and report materials.
- Large `.pt` and `.pth` model checkpoint files.
- Generated demo output directories under `demo_test/test_results/`.

Consequently, the repository contains the uploaded source code, result artifacts, experiment logs, and supporting documentation, but it is **not a completely self-contained runnable package**.

To reproduce training or run inference, you will need to obtain the MSBin dataset, recreate the required data layout, and provide the corresponding trained checkpoints or train the models again.

## 13. Reproducing the Experiments

The following is the general workflow for reproducing the project.

1. Clone or download this repository.
2. Install the Python packages required by the source code.
3. Download the MSBin dataset from the official dataset source.
4. Prepare the dataset and ground-truth masks in the expected layout.
5. Recreate the training, validation, and held-out test page splits.
6. Run the relevant training scripts for the four configurations.
7. Select checkpoints using validation performance.
8. Evaluate the selected checkpoints using the fixed threshold and valid-pixel metric calculations.
9. Compare Precision, Recall, and F1-score across configurations.

Exact command-line arguments and dependency versions should be taken from the corresponding source scripts and any dependency files included in the repository. The dataset splits, preprocessing, and evaluation settings must be kept consistent for a fair comparison.

## 14. Limitations

- Only one training seed was used per configuration.
- The reported differences do not establish statistical significance.
- Band Reweighting and Band Dropping are evaluated only within the tested model and training setup.
- The repository does not include the complete dataset or trained checkpoints.
- The two-page demonstration is illustrative and is not equivalent to the full held-out test evaluation.
- The system performs pixel-level foreground segmentation. It does not perform OCR, text transcription, language understanding, or guaranteed document restoration.

## 15. Future Work

Potential future improvements include:

- Training each configuration with multiple random seeds.
- Reporting mean and standard deviation across independent runs.
- Investigating additional band-selection and spectral-fusion strategies.
- Evaluating performance across more document types and degradation conditions.
- Improving inference speed and memory efficiency.
- Extending the analysis to investigate how performance changes when different spectral bands are unavailable.

## 16. Conclusion

This project investigates multispectral document binarization using a ResNet50-U-Net architecture and two strategies for handling spectral information: Band Reweighting and Band Dropping.

Four model configurations were compared using page-level training, validation, and held-out testing. In the reported test results, Band Reweighting alone achieved the highest F1-score of 0.88318, compared with 0.87347 for the baseline.

The experiments suggest that learning the relative importance of spectral bands can be useful for the tested document binarization task. Further experiments with multiple seeds and additional evaluation settings are needed to establish the reliability and generality of this finding.

## 17. Dataset Reference

MSBin multispectral document dataset.

Dataset DOI: https://doi.org/10.5281/zenodo.3257366

Please consult the dataset's official record for its authorship, citation requirements, terms of use, and download instructions.

---

**Project:** Multispectral Document Binarization Using Band Reweighting and Band Dropping

**Institution:** Amrita Vishwa Vidyapeetham, Amrita School of Computing, Chennai

**Repository:** https://github.com/Koushik153153/Multispectral-Document-Binarization
