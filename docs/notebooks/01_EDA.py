# 01_EDA — Exploratory Data Analysis (script)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path('..').resolve()))
from src.preprocessing import load_data, get_basic_summary

DATA_PATH = Path('../01_data/creditcard.csv')
df = load_data(DATA_PATH)
print('Shape:', df.shape)
get_basic_summary(df)

# Amount distribution
plt.figure(figsize=(10,4))
sns.histplot(df['Amount'], bins=100)
plt.title('Amount distribution')
plt.savefig('../04_results/figures/amount_distribution.png')

# Boxplot by class
plt.figure(figsize=(8,4))
sns.boxplot(x='Class', y='Amount', data=df)
plt.yscale('log')
plt.title('Amount by Class (log scale)')
plt.savefig('../04_results/figures/amount_boxplot.png')

# Hour distribution
df['Hour'] = ((df['Time'] % 86400) // 3600).astype(int)
plt.figure(figsize=(12,4))
sns.countplot(x='Hour', hue='Class', data=df)
plt.title('Transactions by hour')
plt.savefig('../04_results/figures/transactions_by_hour.png')

print('EDA completed. Figures saved to ../04_results/figures/')
