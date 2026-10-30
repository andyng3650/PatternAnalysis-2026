# VQ-VAE2 for Synthetic HipMRI 2D Slice Generation

## 1. Problem and Motivation

## 2. Method
### 2.1 VQ-VAE2 (hierarchical encoder, codebooks, decoder)
### 2.2 PixelCNN Prior (generating new code maps)
### 2.3 Baseline (plain autoencoder, no codebook)

## 3. Feasibility Review

## 4. Data
### 4.1 Dataset
- HipMRI Study 2D slices (keras_slices_data on Rangpur), NIfTI (.nii.gz), float32
- Naming: case_<patient>_week_<scan>_slice_<n>; masks: seg_<patient>_week_<scan>_slice_<n>
### 4.2 Preprocessing
- Per-slice min-max normalisation to [0, 1] (the grayscale channel has no fixed scale, e.g. 0–478)
### 4.3 Split
- Provided split used: 31 / 4 / 3 patients, 11,460 / 660 / 540 slices, 60 slices per scan
- Verified patient-level: zero patient overlap between train, validate and test

## 5. Dependencies

## 6. Usage
### 6.1 Training
### 6.2 Inference / Sampling

## 7. Results
### 7.1 Example Inputs and Outputs
### 7.2 Training Curves
### 7.3 Baseline vs VQ-VAE2
### 7.4 Memorisation Audit (nearest-neighbour analysis)
### 7.5 Resource Profiling

## 8. Failure Autopsies

## 9. Recommendation

## 10. Artificial Intelligence Usage Disclosure

## References