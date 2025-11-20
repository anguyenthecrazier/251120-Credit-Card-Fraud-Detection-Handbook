# Interactive Fraud Detection Demo

Welcome to the **Interactive Fraud Detection Demo** for the  
**Credit Card Fraud Detection – MSc Risk Analytics PLA (Group 6)**.

This page turns our technical work into a **story**:

- See the **imbalance problem** in the dataset  
- Understand the **key EDA insights** that drive model design  
- Compare **unsupervised**, **supervised**, and the final **Hybrid model**  
- See how a single **transaction** is turned into a fraud probability  

> This page is *static* – it shows results exported from our notebooks,
> it does **not** run Python in the browser.

---

## 1. Dataset & Imbalance

We use the classic **Credit Card Fraud Detection** dataset  
(284,807 transactions over 2 days, European cardholders).

| Item | Value |
|------|-------|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Fraud rate | ~0.172% |
| Features | 30 (`Time`, `Amount`, PCA components `V1–V28`) |
| Target | `Class` (0 = legitimate, 1 = fraud) |

!!! warning "Severe class imbalance"
    Only **0.172%** of transactions are fraudulent.  
    A model that predicts *“everything is normal”* gets **99.8% accuracy**  
    but misses **all real frauds** → we care about **recall and F1**, not accuracy.

---

## 2. What EDA Told Us

From the EDA notebook we learned:

- **Fraud amounts**  
  Fraudulent transactions are usually **small “testing” amounts**, with a few **large outliers** where the card is fully exploited.

- **Time-of-day pattern**  
  Fraud is more likely around **~02:00 AM** when normal activity is low,  
  but still appears during the day → fraud is **not** only a “night-time” event.

- **Only 11 truly informative features**  
  Out of 30 features, only **11** have `|corr(Class)| ≥ 0.1`, and none above 0.5.  
  → Fraud patterns are **non-linear and multivariate**, not visible in simple correlations.
  

- **Highly skewed, heavy-tailed features**  
  Many variables are **skewed** and fraud often lies in the **tails**.  
  This motivated **log-transformations** (for anomaly detection) and **tree-based methods** (for supervised learning).

- **Multivariate regression confirms weak linear separability**
  Logistic regression achieves strong ranking performance (ROC-AUC 0.97), but low precision at default thresholds.
  → Fraud patterns exist, but they are subtle, imbalanced, and non-linear, motivating more advanced models.

🔗 Full details in the EDA notebook:  
`../notebooks/EDA_Risk_Analytics_PLA_Credit_Card_Fraud/`

---
## 3. Models & Key Results

We evaluate three categories:

- **Unsupervised anomaly detection**  
- **Supervised machine learning (with SMOTE)**  
- **A final Hybrid meta-model combining all signals**

All performance numbers below are evaluated on the **full original dataset**  
(no SMOTE used in evaluation).

---

## 3.1 Unsupervised Models — Anomaly Detection

From all tested methods, two consistently performed best:

### **Chosen Unsupervised Models**
- **Isolation Forest**
- **Autoencoder**

### **Performance**

| Method | Precision (1) | Recall (1) | F1-score (1) | Accuracy |
|--------|---------------|------------|--------------|----------|
| Isolation Forest (Full) | **0.0294** | **0.9514** | **0.0569** | **0.9512** |
| Autoencoder (Full) | **0.0154** | **0.9014** | **0.0304** | **0.9014** |
| Isolation Forest (High-Corr) | **0.0310** | **0.9515** | **0.0599** | **0.9514** |
| Autoencoder (High-Corr) | **0.0157** | **0.9014** | **0.0309** | **0.9014** |

!!! info "Interpretation"
    - High recall → they catch most frauds  
    - Low precision (typical for anomaly detectors)  
    - Their scores become **powerful Hybrid model inputs**

---

## 3.2 Supervised Models — Learning From Labels (with SMOTE)

Supervised models were trained on **SMOTE-balanced** data  
and evaluated on the **full dataset**.

### **All Supervised Models Tested**

| Model | Precision (1) | Recall (1) | F1-score (1) | Accuracy | Notes |
|-------|---------------|------------|--------------|----------|--------|
| **Random Forest** | **0.9389** | **0.9999** | **0.9685** | **0.9999** | Kept for Hybrid |
| **XGBoost** | **0.8723** | **0.9999** | **0.9318** | **0.9997** | Kept for Hybrid |
| **CatBoost** | **0.8586** | **0.9997** | **0.9239** | **0.9997** | Kept for Hybrid |
| **AdaBoost** | **0.0902** | **0.8951** | **0.1639** | **0.8950** | Excluded |
| **LightGBM** | **0.0094** | **1.0000** | **0.0186** | **0.8735** | Excluded |

!!! failure "Why not AdaBoost or LightGBM?"
    - **AdaBoost**: precision only **0.09** → too many false alarms  
    - **LightGBM**: precision **0.009** despite perfect recall  
    → Both unsuitable for real-world use

---

## 3.3 Hybrid Meta-Model — Final System

The Hybrid model combines **unsupervised anomaly scores**  
and **supervised predicted probabilities**, using:

### **Inputs**
**Unsupervised signals**
- Isolation Forest score (`iso_score`)
- Autoencoder reconstruction error (`ae_mse`)

**Supervised signals**
- Random Forest (`rf_pred`)
- XGBoost (`xgb_pred`)
- CatBoost (`cat_pred`)

### **Meta-classifier**
- **Logistic Regression**
- Threshold optimized for **max F1**
  - ~0.61 (Full data)
  - ~0.597 (High-Corr)

### **Hybrid Performance**

| Metric | Value |
|--------|-------|
| **Precision (1)** | **0.9609** |
| **Recall (1)** | **1.0000** |
| **F1-score (1)** | **0.9801** |
| **Accuracy** | **0.9999** |
| **False Positives** | **20** |
| **False Negatives** | **0** |

!!! success "Why Hybrid Wins"
    - Perfect recall  
    - Best precision  
    - Lowest false positives  
    - Integrates structural anomaly signals + supervised evidence

---

## 4. How the Data & Models Flow (Pipeline)

This section follows the same logic used in the *Models* notebook and the predictive-models poster.

``` 
RAW DATA
(Time, Amount, V1–V28, Class = 0/1, heavily imbalanced)

        ↓

PREPROCESSING
- Log transform skewed features
- MinMaxScaler (for anomaly detection branch)
- SMOTE (for supervised learning branch)

        ↓

UNSUPERVISED LEARNING               SUPERVISED LEARNING
- Isolation Forest (full +          - Random Forest
  high-correlation datasets)        - XGBoost
- Autoencoder (full +               - CatBoost
  high-correlation datasets)

        ↓

HYBRID META-MODEL (Logistic Regression)
- Inputs = unsupervised scores + supervised probabilities

        ↓

FINAL FRAUD PROBABILITY
(Optimised decision threshold ≈ 0.60)
```

### Unsupervised Branch

- **Log-transform highly skewed features**  
  - For each feature (except `Class`), if \|skew\| > 0.75, apply a custom log transform.

- **MinMax scaling**  
  - Scale all features (`Time`, `Amount`, `V1–V28`) into a common range → `X_scaled`.

- **Train anomaly detectors on the full scaled data**  
  - Fit and evaluate on `X_scaled`, `y` using:  
    - Isolation Forest  
    - One-Class SVM  
    - Local Outlier Factor (LOF)  
    - Autoencoder

- **High-correlation subset for improved anomaly detection**  
  - Select only the **11 features** with \|corr(`Class`)\| ≥ 0.1.  
  - Repeat the anomaly-detection experiments on this reduced feature set to see if performance improves.

---

### Supervised Branch

- **Original feature space**  
  - Use the **original** features (no MinMax scaling) for supervised learning.

- **SMOTE oversampling**  
  - Apply **SMOTE** to the full 30-feature dataset  
    → balanced training data: `X_res`, `y_res`.

- **Train–test split on the SMOTE data**  
  - Split into `X_train`, `X_test`, `y_train`, `y_test`.

- **Train supervised models**  
  - Random Forest  
  - XGBoost  
  - CatBoost  
  - AdaBoost  
  - LightGBM

- **Evaluate on the full original dataset**  
  - Use the models trained on SMOTE data to predict on the **full imbalanced dataset** (`df`)  
  - This mimics **real-world deployment**, where fraud is rare.

---

### Hybrid Model

- **Inputs from the Unsupervised Branch (on scaled data)**  
  - `iso_score` = Isolation Forest anomaly score on `X_scaled`  
  - `ae_mse`   = Autoencoder reconstruction error on `X_scaled`  
  - (For the second hybrid variant, the same scores are computed on the 11-feature high-correlation subset.)

- **Inputs from the Supervised Branch (trained on SMOTE, predicted on full data)**  
  - `rf_pred`  = Random Forest P(`Class` = 1)  
  - `xgb_pred` = XGBoost P(`Class` = 1)  
  - `cat_pred` = CatBoost P(`Class` = 1)

- **Build the hybrid feature set**  
  - Stack these features:  
    - `[iso_score, ae_mse, rf_pred, xgb_pred, cat_pred]`

- **Meta-classifier: Logistic Regression**  
  - Train a Logistic Regression model on the **full dataset** using the stacked features.  
  - Choose a decision **threshold ≈ 0.60** to maximise the F1-score.

- **Final output**  
  - Final **fraud probability** and a **fraud / non-fraud** decision.  
  - In the best configuration, the hybrid model achieves **0 false negatives** and only **20 false positives**.

