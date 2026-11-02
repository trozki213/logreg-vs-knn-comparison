# Logistic Regression vs k-Nearest Neighbors

This project compares two classical machine learning algorithms (**k-Nearest Neighbors (k-NN)** and **Logistic Regression**) on a dataset of **1,400 samples × 30 numerical features**.  
The aim is not only to measure and contrast their raw performance, but also to optimize their hyperparameters for the best results in terms of both accuracy and efficiency.  

---

## 📖 Project Outline

We follow a structured pipeline:

1. **Data Loading & Initial Inspection (EDA)**  
   - Overview of dataset shape and class balance  

2. **Correlation Analysis**  
   - Feature correlation matrix and redundancy checks  

3. **Feature Distributions & Scaling**  
   - Standardization of features to ensure fair comparison  

4. **Train/Test Split**  
   - Holdout method for unbiased evaluation  

5. **Baseline Models**  
   - Logistic Regression (linear, interpretable baseline)  
   - k-Nearest Neighbors (non-parametric, flexible decision boundaries)  

6. **Hyperparameter Tuning**  
   - Logistic Regression: regularization parameter `C`  
   - k-NN: number of neighbors `k`, weighting scheme  

7. **Hyperparameter Visualization**  
   - Heatmaps and line plots of performance vs parameters  

8. **Final Model Selection & Evaluation**  
   - Accuracy, precision, recall, F1-score  
   - ROC curves & AUC  

9. **Conclusions**  
   - Trade-offs between interpretability, performance, and efficiency  

---

## 🌍 Motivation

Both **Logistic Regression** and **k-NN** are widely used in practice and form the foundation of many machine learning pipelines.  
This project highlights:  
- How simple algorithms behave on structured data.  
- The role of scaling and hyperparameter tuning.  
- Trade-offs between **interpretability** (LogReg) and **flexibility** (k-NN).

---

## 📊 Results

Evaluated on a stratified 20% hold-out set (280 samples, 5 imbalanced classes); hyperparameters tuned with stratified 10-fold cross-validation on the training set, using weighted F1 as the selection criterion.

| Model | Tuned hyperparameters | CV accuracy | Test accuracy | Test macro F1 |
|---|---|---:|---:|---:|
| Logistic Regression | C = 10, L2 penalty | 0.58 | 0.55 | 0.53 |
| k-NN | k = 9, distance weighting, Euclidean | 0.76 | 0.71 | 0.67 |

k-NN is selected as the final model: its non-linear decision boundaries capture structure that a linear classifier misses, while the rarest class remains the hardest for both models.

## 📂 Repository structure

```
.
├── data/train-data.csv          # 1,400 labelled samples (30 features, 5 classes)
├── data/test-data.csv           # unlabelled features for the final predictions
├── notebook/ML_project.ipynb    # full pipeline, from EDA to final model
└── predictions/predictions.txt  # predicted labels for test-data.csv
```

## ▶️ Running the notebook

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
cd notebook && jupyter notebook ML_project.ipynb
```

