# 04_Hybrid_Model (script)
import pandas as pd
from pathlib import Path
from src.preprocessing import load_data, prepare_supervised_data, prepare_unsupervised_data
from src.hybrid_model import build_meta_features, train_meta_learner
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = load_data('../01_data/creditcard.csv')
X_sup, y = prepare_supervised_data(df)
X_unsup, _ = prepare_unsupervised_data(df)

supervised_models = {
    'rf': RandomForestClassifier(n_estimators=200, random_state=42),
    'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
meta = build_meta_features(X_sup.values, X_unsup.values, y, supervised_models=supervised_models, unsupervised_methods=['isolation_forest','autoencoder'])
print('Meta features shape:', meta.shape)
meta_model, oof = train_meta_learner(meta.drop(columns=['Class']), meta['Class'])
print('Hybrid meta learner trained.')
