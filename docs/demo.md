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
  Many variables are **skewed** and fraud often lives in the **tails**.  
  This motivated **log-transformations** (for anomaly detection) and **tree-based methods** (for supervised learning).

🔗 Full details in the EDA notebook:  
`../notebooks/EDA_Risk_Analytics_PLA_Credit_Card_Fraud/`

---

## 3. Model Highlights

### 3.1 Unsupervised – Anomaly Detection

Train only on **feature patterns**, then flag unusual points.

| Model              | Recall (fraud=1) | Precision (1) | Comment |
|--------------------|------------------|---------------|---------|
| **Isolation Forest** | ~0.85           | ~0.03         | Catches most frauds, but many false alarms |
| **Autoencoder**      | ~0.89           | ~0.02         | Strong recall, good for complex patterns, still low precision |
| **LOF**              | ~0.07           | ~0.003        | Close to random; too many misses and false positives |

**Takeaway:**  
Isolation Forest and Autoencoder are the **only unsupervised models worth keeping**.  
They are later reused as **inputs to the Hybrid model**.

---

### 3.2 Supervised – Learning from Labels (with SMOTE)

Here we **balance the classes with SMOTE**, train on the synthetic-balanced data,  
then **evaluate again on the full original dataset** (no SMOTE).

| Model          | Recall (1) | Precision (1) | F1 (1) | Notes |
|----------------|------------|---------------|--------|-------|
| **Random Forest** | 1.00    | ~0.94        | ~0.97 | 0 false negatives, 32 false positives on full data |
| **XGBoost**       | 1.00    | ~0.87        | ~0.93 | Also perfect recall, slightly more false positives |
| **CatBoost**      | 1.00    | ~0.86        | ~0.92 | Good overall; uses a broader set of features |
| AdaBoost          | ~0.90   | ~0.09        | ~0.16 | Too many false alarms |
| LightGBM          | 1.00    | ~0.01        | ~0.01 | Extremely high false positive rate |

**Takeaway:**  
- **Random Forest** is the **best single supervised model**.  
- XGBoost and CatBoost are also **strong** and kept for the Hybrid.  
- AdaBoost & LightGBM are **not used** in the final system.

---

### 3.3 Hybrid Meta-Model – Final Candidate

The **Hybrid model** combines:

- Unsupervised outputs:  
  - **Isolation Forest** anomaly score (`iso_score`)  
  - **Autoencoder** reconstruction error (`ae_mse`)
- Supervised outputs (trained on **SMOTE-balanced** data, then predicted on the **full** dataset):  
  - **Random Forest**, **XGBoost**, **CatBoost** predicted probabilities (`rf_pred`, `xgb_pred`, `cat_pred`)
- Meta-classifier: **Logistic Regression**, with a threshold chosen to maximise **F1**  
  - Best thresholds ≈ **0.61** (Hybrid 1 – full data) and **0.597** (Hybrid 2 – high-correlated subset)

**Performance on full dataset (Hybrid 1 & 2):**

| Metric (Fraud = 1) | Value |
|--------------------|-------|
| Precision (1)      | 0.9609 |
| Recall (1)         | 1.0000 |
| F1-score (1)       | 0.9801 |
| Accuracy           | 0.9999 |
| False Positives    | 20 |
| False Negatives    | 0 |

**Takeaway:**  
The Hybrid model keeps **perfect recall**, **reduces false positives** further than Random Forest alone, and benefits from **both anomaly signals and supervised scores**.

🔗 Full model details:  
`../notebooks/Models_Risk_Analytics_PLA_Credit_Card_Fraud/`

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


UNSUPERVISED BRANCH
-------------------
1. Log-transform highly skewed features
   • For each feature except Class, if |skew| > 0.75 → apply custom log transform.
2. MinMax scaling
   • Scale all features (Time, V1–V28, Amount) into a common range → X_scaled.
3. Train anomaly detectors on the FULL X_scaled, y:
   • Isolation Forest, One-Class SVM, LOF, Autoencoder.
4. Build high-correlated subset for improved unsupervised models:
   • Keep only 11 features with |corr(Class)| ≥ 0.1 and repeat anomaly detection.

SUPERVISED BRANCH
-----------------
1. Work in the ORIGINAL feature space (no MinMax scaling here).
2. Apply SMOTE on df (all 30 features):
   • Balance Class 0 and Class 1 → X_res, y_res.
3. Train-test split on the SMOTE data:
   • (X_train, X_test, y_train, y_test).
4. Train supervised models:
   • Random Forest, XGBoost, CatBoost, AdaBoost, LightGBM.
5. Re-evaluate trained models on the FULL original dataset df
   • to mimic real-life deployment (heavily imbalanced data).

HYBRID MODEL
------------
1. From the UNSUPERVISED branch (on scaled data):
   • iso_score  = Isolation Forest score on X_scaled
   • ae_mse     = Autoencoder reconstruction error on X_scaled
   • (and, for Model 2, the same but trained on the 11 high-correlated features)

2. From the SUPERVISED branch (trained on SMOTE, predicted on full df):
   • rf_pred   = Random Forest P(Class=1)
   • xgb_pred  = XGBoost P(Class=1)
   • cat_pred  = CatBoost P(Class=1)

3. Stack all these into a hybrid feature set:
   • [iso_score, ae_mse, rf_pred, xgb_pred, cat_pred].

4. Train Logistic Regression meta-model on the full dataset:
   • Choose threshold ≈ 0.60 to maximise F1.

5. Output:
   • Final fraud probability + binary decision
   • Achieves 0 false negatives and only 20 false positives.
