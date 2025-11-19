# Conclusion

The project set out to deepen understanding of real-life fraudulent transaction data and to build and evaluate a comprehensive fraud detection framework. Using the widely adopted Credit Card Fraud Detection dataset, we conducted exploratory data analysis (EDA) and developed multiple predictive models to derive practical insights and inform future work.

## Key Insights from the EDA

The EDA revealed clear behavioural and statistical differences between legitimate and fraudulent transactions:

- **Transaction Amounts:** Fraudulent transactions typically involved small amounts, with occasional high-value outliers. This pattern aligns with known fraud strategies, where small “test” purchases precede larger unauthorized ones.
- **Temporal Patterns:** Fraud was more common during off-peak hours, particularly around 2:00 AM, when legitimate activity is minimal.
- **Correlation Structure:** Only 11 out of 30 features showed an absolute correlation above 0.1 with the fraud label, and none exceeded 0.5. This confirms that simple linear relationships are insufficient to detect fraud, highlighting the need for non-linear, multivariate learning.
- **Feature Distributions:** Most features were non-normal and exhibited distinct ranges for fraudulent activity. This suggested that RobustScaler or StandardScaler were not appropriate preprocessing methods for this dataset.

## Model Performance Overview

We evaluated a range of anomaly detection and machine learning approaches.

### Unsupervised Models

Isolation Forest and Autoencoder achieved high recall scores (0.85 and 0.89), but both suffered from low precision and high false-positive rates. These results indicate that unsupervised methods provide useful signals but require refinement or combination with supervised learning for reliable detection.

### Supervised Models

Supervised algorithms performed substantially better:

- Random Forest achieved an F1-score of 0.9685, perfect recall, and only 32 false positives.
- XGBoost and CatBoost achieved high precision (0.8723 and 0.8585) with zero false negatives.

These results reinforce the strength of supervised approaches when labelled data is available.

### Hybrid Model

The most effective solution was a hybrid ensemble, combining selected supervised and unsupervised model outputs through a Logistic Regression meta-classifier. This configuration produced:

- Accuracy: 0.9999
- Precision: 0.9690
- False Positives: 20

This hybrid structure delivered the strongest balance between precision, recall, and overall reliability.

## Final Remarks

This project highlights the power of combining anomaly detection with machine learning to identify fraudulent financial transactions—especially given the extreme rarity of fraud cases among daily transaction volumes. The insights gained from the EDA and hybrid modelling approach offer promising directions for practical implementation and further research in real-world fraud prevention systems.
