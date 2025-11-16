# 05_Threshold_and_Evaluation (script)
import pandas as pd
from src.evaluation import plot_pr_curve, plot_roc_curve, threshold_selection_table
# This script expects prediction probabilities saved in ../04_results/model_outputs/
try:
    probs = pd.read_csv('../04_results/model_outputs/hybrid_probs_test.csv')
    y_test = pd.read_csv('../04_results/model_outputs/y_test.csv')['Class']
    plot_pr_curve(y_test.values, probs['pred_prob'].values, title='Hybrid PR Curve')
    plot_roc_curve(y_test.values, probs['pred_prob'].values, title='Hybrid ROC Curve')
    thr_table = threshold_selection_table(y_test.values, probs['pred_prob'].values)
    print(thr_table.head())
except Exception as e:
    print('Place model outputs in ../04_results/model_outputs/ to run threshold analysis. Error:', e)
