# Logistic Regression vs k-Nearest Neighbors

This project compares two classical machine learning algorithms — **k-Nearest Neighbors (k-NN)** and **Logistic Regression** — on a dataset of **1,400 samples × 30 numerical features**.  
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
