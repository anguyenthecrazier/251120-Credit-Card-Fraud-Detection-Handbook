# Interactive Demo

# Interactive Fraud Detection Demo

Welcome to the **Interactive Fraud Detection Demo** for the  
**Credit Card Fraud Detection – MSc Risk Analytics PLA (Group 6)**.

This page turns our technical notebook into an **interactive story**:

- Explore the **dataset** and its imbalance problem  
- Compare **unsupervised**, **supervised**, and **hybrid** models  
- See how a transaction would be **flagged as fraud or not**  
- Understand **why** the Hybrid model performs best  

> This page is *static* (runs in the browser only) – it **illustrates** model
> behaviour using real results from our notebooks, but does not execute Python
> live.

---

## 1. Dataset at a Glance

We work with the classic **Credit Card Fraud Detection** dataset  
(284,807 transactions over 2 days, European cardholders).

| Item | Value |
|------|-------|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Fraud rate | ~0.172% |
| Features | 30 (PCA components `V1–V28`, plus `Time`, `Amount`) |
| Target | `Class` (0 = legitimate, 1 = fraud) |

!!! warning "Severe class imbalance"
    Only **0.172%** of transactions are fraudulent.  
    A naïve model that predicts *“everything is normal”* reaches **99.8% accuracy**  
    but misses **all real frauds**.  
    → This is why we focus on **recall** and **F1-score**, not accuracy alone.

---

### 1.1 Key Behavioural Patterns

!!! info "What we learned from EDA"
    - **Fraud Amounts**: Many frauds are **small “testing” amounts** with a few
      very **high-value spikes**.
    - **Time-of-day effect**: Fraud peaks around **02:00 AM**, when genuine
      activity is low – but also appears during normal business hours.
    - **PCA features**: Only **11/30** features have a correlation with `Class`
      above 0.1, and none above 0.5 → fraud is **complex and non-linear**.
    - **Distributions**: Many features are skewed and heavy-tailed  
      → simple linear models and standard scalers are not sufficient.

You can see all EDA details in the notebook:

- 📓 **EDA Notebook**: `notebooks/eda/EDA_Risk_Analytics_PLA_Credit_Card_Fraud.ipynb`

---

## 2. Choose a Model

Use the tabs below to explore **how each family of models behaves**.  
Numbers and plots come from our modelling notebook.

> 📓 **Modelling Notebook**:  
> `notebooks/modelling/Risk_Analytics_PLA_Credit_Card_Fraud.ipynb`

---

### 2.1 Unsupervised Models – Anomaly Detectors

These models **do not need labels** during training.  
They learn what “normal” looks like and flag deviations as **anomalies**.

=== "Isolation Forest"

Isolation Forest isolates anomalies by randomly splitting the data.

**Why we use it**

- Scales well to large datasets  
- Good at detecting “weird” points in high dimensions  

**Performance (full dataset)**

| Metric (Fraud = Class 1) | Value |
|--------------------------|-------|
| Precision (1) | 0.029 |
| Recall (1) | 0.852 |
| F1-score (1) | 0.057 |
| Accuracy | 0.951 |

**Intuition**

- Detects **most frauds** (high recall)  
- But precision is low → many **false alarms**  

![Isolation Forest Confusion Matrix](../images/iforest_cm.png){ width=400 }
![Isolation Forest ROC](../images/iforest_roc.png){ width=400 }

*(Replace the image paths with your actual plot files.)*

---

=== "Autoencoder"

An **Autoencoder** compresses and reconstructs normal transactions.  
Frauds are detected when the **reconstruction error is high**.

**Performance (full dataset)**

| Metric (Fraud = Class 1) | Value |
|--------------------------|-------|
| Precision (1) | 0.015 |
| Recall (1) | 0.894 |
| F1-score (1) | 0.030 |
| Accuracy | 0.901 |

**Strengths**

- Very good at capturing **complex, non-linear patterns**  
- High recall: detects most frauds  

**Weaknesses**

- Very low precision → large manual review workload  
- Needs careful architecture / threshold tuning  

![Autoencoder Confusion Matrix](../images/autoencoder_cm.png){ width=400 }

---

=== "Local Outlier Factor (LOF)"

LOF compares the **local density** of each point with its neighbours.

- In theory good for **local anomalies**
- In practice here, performance is **poor** due to:
  - high dimensionality
  - strong imbalance
  - overlapping distributions

| Metric (Fraud = Class 1) | Value |
|--------------------------|-------|
| Precision (1) | 0.003 |
| Recall (1) | 0.071 |
| F1-score (1) | 0.005 |
| Accuracy | 0.949 |

![LOF ROC](../images/lof_roc.png){ width=400 }

> LOF performs **close to random guessing** in this dataset and is **not**
> selected for the final hybrid system.

---

### 2.2 Supervised Models – Learning from Labels

These models train on **labelled fraud vs non-fraud** transactions  
(using **SMOTE** oversampling for the minority class).

=== "Random Forest"

Random Forest is our **best standalone supervised model**.

| Metric (Fraud = Class 1) | Value |
|--------------------------|-------|
| Precision (1) | 0.939 |
| Recall (1) | 1.000 |
| F1-score (1) | 0.969 |
| Accuracy | 0.9999 |

- No **false negatives** on the test set  
- Only **32 false positives** out of 284,807 transactions  
- Provides useful **feature importance** rankings  

![Random Forest Confusion Matrix](../images/rf_cm.png){ width=400 }
![Random Forest ROC](../images/rf_roc.png){ width=400 }

---

=== "XGBoost"

Gradient-boosted trees with strong performance on imbalanced data.

| Metric (Fraud = Class 1) | Value |
|--------------------------|-------|
| Precision (1) | 0.872 |
| Recall (1) | 1.000 |
| F1-score (1) | 0.932 |
| Accuracy | 0.9997 |

- Also achieves **zero false negatives**  
- Slightly more false positives than Random Forest

![XGBoost Confusion Matrix](../images/xgb_cm.png){ width=400 }

---

=== "CatBoost"

Handles categorical-like behaviour and complex interactions well.

| Metric (Fraud = Class 1) | Value |
|--------------------------|-------|
| Precision (1) | 0.859 |
| Recall (1) | 1.000 |
| F1-score (1) | 0.924 |
| Accuracy | 0.9997 |

![CatBoost Confusion Matrix](../images/catboost_cm.png){ width=400 }

---

=== "AdaBoost & LightGBM"

These models underperformed on this dataset:

| Model | Precision (1) | Recall (1) | F1-score (1) | Notes |
|-------|---------------|-----------|--------------|-------|
| AdaBoost | 0.088 | 0.900 | 0.160 | Many false positives |
| LightGBM | 0.007 | 1.000 | 0.014 | Extremely high false positive rate |

> We **do not include** these models in the final hybrid system  
> due to their **poor precision**.

---

### 2.3 Hybrid Model – Best of Both Worlds

The final production candidate is a **Hybrid Model** that combines:

- Unsupervised: **Isolation Forest**, **Autoencoder**  
- Supervised: **Random Forest**, **XGBoost**, **CatBoost**  
- Meta-classifier: **Logistic Regression**

Each base model outputs a score, and the meta-classifier learns how to
**optimally weight** them.

| Metric (Fraud = Class 1) | Value |
|--------------------------|-------|
| Precision (1) | 0.961 |
| Recall (1) | 1.000 |
| F1-score (1) | 0.980 |
| Accuracy | 0.9999 |
| False Positives | 20 |
| False Negatives | 0 |

![Hybrid Confusion Matrix](../images/hybrid_cm.png){ width=400 }
![Hybrid PR Curve](../images/hybrid_pr.png){ width=400 }

!!! success "Why the Hybrid Model is preferred"
    - Keeps the **perfect recall** of the best supervised models  
    - **Reduces false positives** compared with Random Forest alone  
    - Adds **anomaly-detection signals** from unsupervised models  
    - More robust to **new, unseen fraud patterns**

---

## 3. How the Hybrid Pipeline Works

```mermaid
flowchart TD
    A[Raw Transaction] --> B[Feature Preprocessing<br/>Log transform, MinMax scaling]
    B --> C1[Isolation Forest]
    B --> C2[Autoencoder]
    B --> C3[Random Forest]
    B --> C4[XGBoost]
    B --> C5[CatBoost]
    C1 --> D[Logistic Regression<br/>Meta-Classifier]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    D --> E[(Final Fraud Probability)]
