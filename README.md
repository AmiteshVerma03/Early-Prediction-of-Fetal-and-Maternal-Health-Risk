**Project Overview**
- **Name:**: Multimodal Fetal & Maternal Health Prediction System
- **Description:**: End-to-end multimodal machine learning project that predicts fetal and maternal health risk levels using CTG signals, ECG-HRV features, ultrasound numeric parameters, ultrasound images (CNN), and maternal clinical tabular features. The repository demonstrates preprocessing pipelines, classical ML, deep learning, and optional multimodal fusion.

**Repository Structure**
- **Notebooks:**: `01_ctg_tabular.ipynb`, `02_ecg_hrv.ipynb`, `03_ultrasound_numeric.ipynb`, `03_ultrasound_resnet.ipynb`, `04_maternal_tabular.ipynb`, `99_optional_fusion.ipynb`
- **Data (expected):**: `data/Ultrasound Fetus Dataset/` and modality-specific folders used by the notebooks.
- **Model artifacts:**: `model_artifacts/` (trained models saved from notebooks)

**Notebooks & Purpose**
- **01_ctg_tabular.ipynb:**: CTG (cardiotocography) tabular features, preprocessing, and classical ML models (Logistic Regression, Random Forest, XGBoost, SVM). Includes evaluation and feature importance visualizations.
- **02_ecg_hrv.ipynb:**: ECG-based HRV extraction (time & frequency domain) and classification experiments.
- **03_ultrasound_numeric.ipynb:**: Tabular ultrasound biometric features — cleaning, scaling, feature engineering and ML models.
- **03_ultrasound_resnet.ipynb:**: Image-based modeling using transfer learning (ResNet50). Data augmentation, training/validation pipelines and model evaluation.
- **04_maternal_tabular.ipynb:**: Maternal clinical tabular model building and evaluation.
- **99_optional_fusion.ipynb:**: Optional multimodal fusion (feature-level or prediction-level) and stacking/ensemble strategies.

**Quick Setup**
- **Python version:**: Recommended `3.10` or newer.
- **Create virtual environment (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- **Install dependencies (suggestion):**
```powershell
pip install -r requirements.txt
```

**Suggested `requirements.txt` (minimal)**
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`  # for CNNs and ResNet50
- `keras`        # optional (if separate)
- `matplotlib`
- `seaborn`
- `wfdb`         # if using PhysioNet CTG signal utilities
- `neurokit2`    # optional for ECG/HRV
- `xgboost`

(Adjust versions for reproducibility and GPU support — e.g., `tensorflow-gpu` if appropriate.)

**Data Preparation**
- Place each modality's dataset in the `data/` folder following the notebook-specific expected layout. For example, the ultrasound image notebook expects a directory with `train/`, `validation/`, and `test/` subfolders containing class folders.
- If CTG or ECG datasets are large, reference them externally and add a small sample dataset for quick local runs.

**How to Run Notebooks**
- Open the repository in VS Code or Jupyter Lab and run each notebook in order for the modality you want to test.
- Typical workflow:
  - Prepare data and place under `data/`
  - Update any absolute paths in the notebooks to match your environment (or change to relative paths)
  - Run preprocessing cells, then model training cells

**Reproducible Training Tips**
- Use fixed random seeds for train/test splits: `random_state=42` in `train_test_split`.
- Use `ImageDataGenerator` with `color_mode='rgb'` when using ImageNet pretrained models like ResNet50.
- If using a single-channel input, either convert to 3 channels or set `weights=None` when creating the base model.

**Improving Performance**
- Data augmentation (rotation, shifts, zoom, brightness, flips)
- Two-phase training: train top layers, then unfreeze and fine-tune with a lower learning rate
- Use `ReduceLROnPlateau` and `EarlyStopping` callbacks
- Apply class weights if classes are imbalanced
- Experiment with ensembling and stacking in `99_optional_fusion.ipynb`

**Saving & Loading Models**
- CNN models are saved in `model_artifacts/` as `.h5` files by default in notebooks. Example:
```python
model.save(r"model_artifacts\fetal_ultrasound_classifier.h5")
```

**Next Steps & Recommendations**
- Add a `requirements.txt` file with pinned versions for reproducibility.
- Add a small `sample_data/` subset for quick runs and continuous integration.
- Consider adding a `scripts/` folder for data-prep helpers and a `Makefile` or PowerShell scripts for common tasks.

**License & Contributing**
- Add a `LICENSE` file if you want to publish the project.
- Add `CONTRIBUTING.md` to describe how others can contribute to dataset cleaning, model training, or evaluation.

**Contact**
- For questions or improvements, open issues or PRs in the repository.

---
*README generated automatically. If you want, I can also create a `requirements.txt`, add a small `sample_data/` scaffold, or update any notebook paths to use relative paths.*
