"""
Name: Phase 1A: Load & Clean Datasets
Description: Loads COMPAS recidivism and Adult income datasets from public sources, applies ProPublica-standard filters and cleaning (removing missing values, outliers, invalid entries), selects key features, and exports cleaned datasets as CSV files for fairness analysis.
"""


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# COMPAS RECIDIVISM DATASET
# Source: ProPublica GitHub
# ─────────────────────────────────────────────
compas_url = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)
compas_raw = pd.read_csv(compas_url)

# Filter to standard research subset (as per ProPublica methodology)
compas_raw = compas_raw[
    (compas_raw['days_b_screening_arrest'] <= 30) &
    (compas_raw['days_b_screening_arrest'] >= -30) &
    (compas_raw['is_recid'] != -1) &
    (compas_raw['c_charge_degree'] != 'O') &
    (compas_raw['score_text'] != 'N/A')
].copy()

# Select relevant columns
compas_cols = [
    'age', 'c_charge_degree', 'race', 'sex', 'priors_count',
    'days_b_screening_arrest', 'decile_score', 'is_recid',
    'two_year_recid', 'length_of_stay'
]
# Compute length_of_stay safely
compas_raw['length_of_stay'] = (
    pd.to_datetime(compas_raw['c_jail_out'], errors='coerce') -
    pd.to_datetime(compas_raw['c_jail_in'], errors='coerce')
).dt.days.fillna(0).clip(lower=0)

compas_df = compas_raw[compas_cols].copy()
compas_df = compas_df.dropna()

# Sensitive attributes
compas_sensitive = ['race', 'sex']
compas_target = 'two_year_recid'

print("=" * 60)
print("COMPAS RECIDIVISM DATASET")
print("=" * 60)
print(f"Shape          : {compas_df.shape}")
print(f"Target         : {compas_target}")
print(f"Sensitive Attrs: {compas_sensitive}")
print(f"\nClass Balance:")
_vc = compas_df[compas_target].value_counts()
print(f"  No Recidivism (0): {_vc.get(0, 0):,} ({_vc.get(0,0)/len(compas_df)*100:.1f}%)")
print(f"  Recidivism    (1): {_vc.get(1, 0):,} ({_vc.get(1,0)/len(compas_df)*100:.1f}%)")
print(f"\nRace Distribution:")
for race, cnt in compas_df['race'].value_counts().items():
    print(f"  {race:<25} {cnt:>5} ({cnt/len(compas_df)*100:.1f}%)")
print(f"\nSex Distribution:")
for sex, cnt in compas_df['sex'].value_counts().items():
    print(f"  {sex:<25} {cnt:>5} ({cnt/len(compas_df)*100:.1f}%)")
print(f"\nMissing Values: {compas_df.isnull().sum().sum()}")
print(f"\nDtypes:\n{compas_df.dtypes}")

# ─────────────────────────────────────────────
# ADULT INCOME DATASET
# Source: UCI ML Repository
# ─────────────────────────────────────────────
adult_cols = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week',
    'native_country', 'income'
]
adult_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
adult_df = pd.read_csv(adult_url, header=None, names=adult_cols,
                        na_values=' ?', skipinitialspace=True)

# Clean
adult_df = adult_df.dropna()
adult_df['income'] = (adult_df['income'].str.strip()
                      .str.replace('.', '', regex=False)
                      .map({'<=50K': 0, '>50K': 1}))
adult_df = adult_df.drop(columns=['fnlwgt'])  # weight column not useful

adult_sensitive = ['race', 'sex']
adult_target = 'income'

print("\n" + "=" * 60)
print("ADULT INCOME DATASET")
print("=" * 60)
print(f"Shape          : {adult_df.shape}")
print(f"Target         : {adult_target}")
print(f"Sensitive Attrs: {adult_sensitive}")
print(f"\nClass Balance:")
_vc2 = adult_df[adult_target].value_counts()
print(f"  Income <=50K (0): {_vc2.get(0, 0):,} ({_vc2.get(0,0)/len(adult_df)*100:.1f}%)")
print(f"  Income  >50K (1): {_vc2.get(1, 0):,} ({_vc2.get(1,0)/len(adult_df)*100:.1f}%)")
print(f"\nSex Distribution:")
for sex, cnt in adult_df['sex'].value_counts().items():
    print(f"  {sex:<25} {cnt:>5} ({cnt/len(adult_df)*100:.1f}%)")
print(f"\nRace Distribution:")
for race, cnt in adult_df['race'].value_counts().items():
    print(f"  {race:<25} {cnt:>5} ({cnt/len(adult_df)*100:.1f}%)")
print(f"\nMissing Values: {adult_df.isnull().sum().sum()}")

# Save to CSV
compas_df.to_csv('compas_cleaned.csv', index=False)
adult_df.to_csv('adult_cleaned.csv', index=False)
print("\n✅ Saved: compas_cleaned.csv")
print("✅ Saved: adult_cleaned.csv")
print("\n✅ PHASE 1A COMPLETE — Both datasets loaded and cleaned.")
