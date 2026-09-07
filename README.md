# Lightweight Multimodal Parkinson's Disease Detection (MoSVM)

Multimodal Parkinson's disease (PD) screening from voice recordings, spiral handwriting drawings, and structural MRI. Each modality is classified independently by its own SVM; predictions are combined via weighted soft-voting late fusion, which keeps the system usable when one or more modalities are missing.

## Repository structure

```
notebooks/
  new_dataset_preprocessing.ipynb   MRI (TaoWu): preprocessing, feature extraction, GroupKFold CV, hyperparameter search
  voice_grouped_cv.ipynb            Voice (UCI): preprocessing, feature selection, LOSO CV, hyperparameter search
  spiral_grouped_cv.ipynb           Spiral (kmader): MobileNetV2 features, LOSO CV, hyperparameter search
  fusion_ablation.ipynb             Ablation + fusion table across all modality combinations
  weight_sense.ipynb          Grid search over soft-vote weights
  final_system.ipynb                Final models trained on all data + MoSVM inference function + honest result figures

results/
  results_voice.csv                 Out-of-fold subject-level predictions (voice)
  results_spiral.csv                Out-of-fold subject-level predictions (spiral)
  results_mri.csv                   Out-of-fold subject-level predictions (MRI)
  ablation_fusion_table.csv         Per-modality and per-combination metrics
  weight_sensitivity_table.csv      Metrics across the weight grid
```

## Data setup

Three public datasets are used. None of the raw data is committed to this repo;

**Voice — UCI Parkinson's Voice Dataset.** No manual download needed. `voice_grouped_cv.ipynb` and `final_system.ipynb` fetch it automatically at runtime via `urllib.request` from `archive.ics.uci.edu`.

**Spiral — Parkinson's Drawings Dataset (Kaggle, kmader).** Download from Kaggle and place the zip so the notebook can find it, or extract it yourself into `data/parkinsons_drawings/` with the structure:
```
parkinsons_drawings/
  spiral/training/{healthy,parkinson}/
  spiral/testing/{healthy,parkinson}/
```

**MRI — TaoWu dataset (T1-weighted, BIDS format).** Available via NITRC / the Neurocon-TaoWu release referenced in Badea et al. (2017). Place under `data/taowu/` in BIDS layout:
```
taowu/
  sub-*/anat/*_T1w.nii.gz
```

**Note on paths:** the notebooks currently use relative paths (`../parkinsons.data`, `parkinsons_drawings`, `taowu`) inherited from how they were originally run, not `data/`. Either adjust these three path variables at the top of each notebook to point into `data/`, or mirror the original flat layout — either works, just be consistent.

## Requirements

Python 3.10+, with: `numpy`, `pandas`, `scikit-learn`, `tensorflow` (MobileNetV2/Keras), `nibabel`, `scipy`, `Pillow`, `matplotlib`. GPU is not required; all reported benchmarks are CPU-only.

## Run order

1. `voice_grouped_cv.ipynb`, `spiral_grouped_cv.ipynb`, `new_dataset_preprocessing.ipynb` — run independently, in any order. Each writes its `results_*.csv`.
2. `fusion_ablation.ipynb` — reads all three `results_*.csv`, writes `ablation_fusion_table.csv`.
3. `weight_sensitivity.ipynb` — reads all three `results_*.csv`, writes `weight_sensitivity_table.csv`.
4. `final_system.ipynb` — trains the final deployable models on full data, defines the fusion inference function, and generates the reported figures from `results_*.csv` and `ablation_fusion_table.csv`.

## Methodology notes

- **Cross-validation is subject-level, not sample-level.** Leave-One-Subject-Out for voice (32 subjects) and spiral (28 subjects); 5-fold group k-fold for MRI (40 subjects). Multiple samples per subject (voice recordings, MRI slices) never span both train and test within a fold.
- **Hyperparameters** (SVM `C`, `gamma`, `kernel`) are selected via grid search nested inside each outer fold, using a group-aware inner split on training subjects only.
- **Fusion is evaluated via Monte Carlo synthetic pairing.** The three datasets are independent cohorts with no shared patients, so multi-modality rows in `ablation_fusion_table.csv` are computed by randomly pairing one subject's out-of-fold probability from each modality (matched on label) and averaging, repeated 500 times. This illustrates the fusion mechanism; it is not validation on real multi-modal patients.

## Data citations

- Little, M.: Parkinsons. UCI Machine Learning Repository (2007). https://doi.org/10.24432/C59C74
- Zham, P., Kumar, D.K., Dabnichki, P., Arjunan, S.P., Raghav, S.: Distinguishing different stages of Parkinson's disease using composite index of speed and pen-pressure of sketching a spiral. Frontiers in Neurology 8 (2017). https://doi.org/10.3389/fneur.2017.00435
- Badea, L., Onu, M., Wu, T., Roceanu, A., Bajenaru, O.: Exploring the reproducibility of functional connectivity alterations in Parkinson's disease. PLoS ONE 12(11), e0188196 (2017). https://doi.org/10.1371/journal.pone.0188196
