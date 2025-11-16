# 02_Unsupervised_Models (script)
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from src.preprocessing import load_data, prepare_unsupervised_data
from src.evaluation import evaluate_binary_preds
Path('../04_results/figures').mkdir(parents=True, exist_ok=True)

df = load_data('../01_data/creditcard.csv')
X_unsup, y = prepare_unsupervised_data(df)

# Isolation Forest
contamination = max(0.001, y.sum()/len(y))
if_clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
if_clf.fit(X_unsup)
scores_if = -if_clf.decision_function(X_unsup)
thr = np.percentile(scores_if, 100*(1 - y.sum()/len(y)))
y_pred_if = (scores_if >= thr).astype(int)
evaluate_binary_preds(y, y_pred_if, label='IsolationForest')
