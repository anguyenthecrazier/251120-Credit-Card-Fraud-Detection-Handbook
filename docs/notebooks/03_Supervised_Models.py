# 03_Supervised_Models (script)
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.preprocessing import load_data, prepare_supervised_data
from src.evaluation import evaluate_model_on_test

df = load_data('../01_data/creditcard.csv')
X, y = prepare_supervised_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print('After SMOTE:', np.bincount(y_train_sm))

rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train_sm, y_train_sm)
probs_rf, preds_rf = evaluate_model_on_test(rf, X_test, y_test, label='RandomForest')

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_sm, y_train_sm)
evaluate_model_on_test(xgb, X_test, y_test, label='XGBoost')
