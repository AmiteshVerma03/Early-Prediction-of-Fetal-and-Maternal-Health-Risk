📌 Multimodal Fetal & Maternal Health Prediction System

This project is a complete end-to-end Multimodal Machine Learning Pipeline designed to predict fetal and maternal health risks by integrating tabular clinical data, CTG data, ECG-HRV signals, ultrasound numeric features, and ultrasound images using both classical ML and Deep Learning.

The repository is organized into multiple Jupyter notebooks, each representing a specific module in the pipeline.

🚀 📁 Project Structure
📦 Multimodal-Fetal-Maternal-Health-Prediction
│
├── 01_ctg_tabular.ipynb
├── 02_ecg_hrv.ipynb
├── 03_ultrasound_numeric.ipynb
├── 03_ultrasound_resnet.ipynb
├── 04_maternal_tabular.ipynb
└── 99_optional_fusion.ipynb

🧠 Project Overview

This project aims to identify fetal or maternal health risks early by combining multiple physiological sources:

✔ CTG (Cardiotocography)
✔ ECG-HRV (Heart Rate Variability Signals)
✔ Ultrasound Numeric Data
✔ Ultrasound Imaging (Deep Learning via ResNet)
✔ Maternal Clinical Data (Tabular)

The system uses both traditional ML models (XGBoost, Random Forest, Logistic Regression, SVM) and Deep Learning models (CNN, ResNet50) to generate predictions from various modalities.

Finally, the optional fusion module can combine embeddings/features from multiple models to produce a single unified health-risk prediction.

📚 Notebook Descriptions
📘 01_ctg_tabular.ipynb – CTG-Based Risk Classification

This module processes CTG signals and performs classification using multiple ML models.

🔹 Steps covered:

Data loading & cleaning

Feature normalization

Outlier handling

Model training:

Logistic Regression

Random Forest

XGBoost

SVM

Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC

Feature importance visualization

📘 02_ecg_hrv.ipynb – ECG & Heart Rate Variability Analysis

This notebook extracts physiological HRV features and builds ML-based classifiers.

🔹 Highlights:

RMSSD, SDNN, LF/HF, pNN50 feature extraction

Statistical time-domain and frequency-domain analysis

ML models:

Random Forest

Gradient Boosting

XGBoost

Performance comparison

HRV feature importance ranking

📘 03_ultrasound_numeric.ipynb – Ultrasound Measurements (Numeric)

This module handles numeric ultrasound measurements (fetal length, head circumference, etc.).

🔹 Includes:

Missing value handling

Normalization

Outlier cleanup

ML models (RF, XGB, LR, SVM)

Classification accuracy & ROC-AUC

Correlation heatmap + PCA visualization

📘 03_ultrasound_resnet.ipynb – Ultrasound Image Classification (Deep Learning)

This is the CNN-based ultrasound image classifier using transfer learning.

🔹 Features:

Data augmentation (rotation, shift, zoom)

Preprocessing & loading from directory

Transfer learning using ResNet50

GlobalAveragePooling + Dense head

Accuracy, precision, recall, F1-score

Training/validation curves

Class-wise prediction performance

📘 04_maternal_tabular.ipynb – Maternal Health Risk Prediction

This notebook uses maternal clinical tabular data (BP, glucose, age, hemoglobin, etc.).

🔹 Covers:

Data cleaning

Scaling

Train-test split

Classical ML models

Confusion matrices

Health risk classification (Low/Medium/High risk)

📘 99_optional_fusion.ipynb – Multimodal Fusion (Optional Module)

This is an advanced notebook that attempts to fuse outputs/embeddings from:

CTG Model

ECG-HRV Model

Ultrasound Numeric Model

Ultrasound ResNet Image Model

Maternal Tabular Model

🔹 Combines:

Concatenation-based fusion

Ensemble voting

Weighted averaging

Stacking classifier (meta-model)

🧪 Models Used in This Project
✔ Classical ML Models

XGBoost

Random Forest

Decision Tree

Logistic Regression

SVM

✔ Deep Learning Models

ResNet50 (transfer learning)

CNN classifier (custom)

📈 Evaluation Metrics

For all ML/DL models:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

For CNN models:

Loss curves

Validation accuracy curves

Per-class accuracy

📊 Visualizations Included

Correlation heatmaps

Feature importance plots

PCA visualizations

ROC curves

Training/validation curves

Confusion matrices

💡 Key Contributions of This Project
🔸 Built complete data pipelines for 5 different modalities
🔸 Implemented multiple ML models on clinical & physiological datasets
🔸 Built a CNN image classifier using ResNet50
🔸 Explored multimodal fusion to enhance prediction accuracy
🔸 Generated detailed visualizations for all processed datasets
🏁 Outcome

This project is a full pipeline demonstrating:

Medical data preprocessing

Feature engineering

Model building

Deep learning for image diagnosis

Multimodal fusion

End-to-end evaluation

Perfect for academic research, ML portfolios, or interview demonstrations.
