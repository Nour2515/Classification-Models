# 🧠 Machine Learning Projects: House Price & Breast Cancer Diagnosis

This repository includes two supervised machine learning projects:

1. 🏠 **House Price Classification** using KNN with preprocessing, visualization, and cross-validation.
2. 🔬 **Breast Cancer Diagnosis** using SVM and Neural Networks (MLP), including grid search and loss curve analysis.

---

## 📁 Datasets Used

### 1. House Price Dataset
- `house_price_regression_dataset.csv`
- Features: `Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Neighborhood_Quality`, `Lot_Size`, `Garage_Size`
- Target: `House_Price` (classified into `low`, `medium`, `high` using binning)

### 2. Breast Cancer Dataset
- `data.csv` from the UCI ML Repository
- Binary classification: `Malignant (M)` vs `Benign (B)`
- Target: `diagnosis`

---

## 🏠 House Price Classification (KNN)

### 📊 Data Preprocessing
- Labeling house prices into 3 classes using `pd.cut()`
- Feature scaling using `StandardScaler`
- Train/Test/Validation splitting (60/20/20)

### 🔍 Model Selection
- K-Nearest Neighbors (KNN)
- Best `k` selected using validation accuracy

```python
for k in range(1, 30):
    ...
    accuracy_score(y_val, y_pred)
