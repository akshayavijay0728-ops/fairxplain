"""
Name: Phase 1B: Feature Preprocessing & Splits
Description: Preprocesses COMPAS and Adult Income datasets for fairness analysis: applies feature engineering (age binning, encoding), removes sensitive attributes (race/sex) from features, scales/encodes features, performs stratified 70/30 train-test splits, and exports metadata.
"""


import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# ─────────────────────────────────────────────
# COMPAS PREPROCESSING
# ─────────────────────────────────────────────
compas = compas_df.copy()

# Feature engineering: age bins
compas['age_bin'] = pd.cut(compas['age'],
    bins=[0, 25, 35, 45, 55, 100],
    labels=['<25', '25-35', '35-45', '45-55', '55+'])

# Encode c_charge_degree: M=0, F=1
compas['charge_degree_enc'] = (compas['c_charge_degree'] == 'F').astype(int)

# Define features (NO sensitive attributes in X)
compas_feature_cols = [
    'age', 'charge_degree_enc', 'priors_count',
    'days_b_screening_arrest', 'length_of_stay'
]
compas_cat_cols = []
compas_num_cols = compas_feature_cols

# Sensitive attribute arrays (preserved separately)
compas_race = compas['race'].values
compas_sex  = compas['sex'].values

X_compas = compas[compas_feature_cols].values.astype(float)
y_compas = compas[compas_target].values

# 70/30 stratified split
(X_compas_train, X_compas_test,
 y_compas_train, y_compas_test,
 race_compas_train, race_compas_test,
 sex_compas_train, sex_compas_test) = train_test_split(
    X_compas, y_compas, compas_race, compas_sex,
    test_size=0.30, random_state=RANDOM_STATE,
    stratify=y_compas
)

# Scale features
scaler_compas = StandardScaler()
X_compas_train = scaler_compas.fit_transform(X_compas_train)
X_compas_test  = scaler_compas.transform(X_compas_test)

print("=" * 60)
print("COMPAS PREPROCESSING COMPLETE")
print("=" * 60)
print(f"Feature columns : {compas_feature_cols}")
print(f"Train size      : {X_compas_train.shape}")
print(f"Test  size      : {X_compas_test.shape}")
print(f"Train class dist: {np.bincount(y_compas_train)}")
print(f"Test  class dist: {np.bincount(y_compas_test)}")

# ─────────────────────────────────────────────
# ADULT INCOME PREPROCESSING
# ─────────────────────────────────────────────
adult = adult_df.copy()

# Feature engineering: age bins
adult['age_bin'] = pd.cut(adult['age'],
    bins=[0, 25, 35, 45, 55, 100],
    labels=['<25', '25-35', '35-45', '45-55', '55+'])

# Encode sex binary
adult['sex_enc'] = (adult['sex'] == 'Male').astype(int)

# Define features (NO race or sex in X)
adult_num_cols  = ['age', 'education_num', 'capital_gain',
                   'capital_loss', 'hours_per_week']
adult_cat_cols  = ['workclass', 'education', 'marital_status',
                   'occupation', 'relationship', 'native_country']

adult_race = adult['race'].values
adult_sex  = adult['sex'].values

y_adult = adult[adult_target].values

# Build preprocessor
preprocessor_adult = ColumnTransformer(transformers=[
    ('num', StandardScaler(), adult_num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), adult_cat_cols)
])

X_adult_raw = adult[adult_num_cols + adult_cat_cols]

# 70/30 stratified split
(X_adult_train_raw, X_adult_test_raw,
 y_adult_train, y_adult_test,
 race_adult_train, race_adult_test,
 sex_adult_train, sex_adult_test) = train_test_split(
    X_adult_raw, y_adult, adult_race, adult_sex,
    test_size=0.30, random_state=RANDOM_STATE,
    stratify=y_adult
)

# Fit preprocessor on train only (no leakage)
X_adult_train = preprocessor_adult.fit_transform(X_adult_train_raw)
X_adult_test  = preprocessor_adult.transform(X_adult_test_raw)

# Feature names after one-hot encoding
adult_ohe_cols = (preprocessor_adult.named_transformers_['cat']
                  .get_feature_names_out(adult_cat_cols).tolist())
adult_feature_names = adult_num_cols + adult_ohe_cols

print("\n" + "=" * 60)
print("ADULT INCOME PREPROCESSING COMPLETE")
print("=" * 60)
print(f"Numeric features: {adult_num_cols}")
print(f"Categorical cols: {adult_cat_cols}")
print(f"Total features  : {X_adult_train.shape[1]}")
print(f"Train size      : {X_adult_train.shape}")
print(f"Test  size      : {X_adult_test.shape}")
print(f"Train class dist: {np.bincount(y_adult_train)}")
print(f"Test  class dist: {np.bincount(y_adult_test)}")

# Save metadata
meta = {
    'compas': {
        'feature_cols': compas_feature_cols,
        'target': 'two_year_recid',
        'sensitive': ['race', 'sex'],
        'train_size': int(X_compas_train.shape[0]),
        'test_size': int(X_compas_test.shape[0]),
        'n_features': int(X_compas_train.shape[1])
    },
    'adult': {
        'num_cols': adult_num_cols,
        'cat_cols': adult_cat_cols,
        'total_features': int(X_adult_train.shape[1]),
        'target': 'income',
        'sensitive': ['race', 'sex'],
        'train_size': int(X_adult_train.shape[0]),
        'test_size': int(X_adult_test.shape[0])
    }
}
with open('preprocessing_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n✅ Saved: preprocessing_metadata.json")
print("✅ PHASE 1B COMPLETE — Preprocessing and splits ready.")
