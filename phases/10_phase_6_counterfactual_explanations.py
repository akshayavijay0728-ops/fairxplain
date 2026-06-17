"""
Name: Phase 6: Counterfactual Explanations
Description: Generates counterfactual explanations using DiCE for COMPAS recidivism and Adult income models. Shows what features would need to change to flip unfavorable predictions (e.g., from recidivist to non-recidivist) while keeping sensitive attributes immutable. Visualizes original vs. counterfactual feature values.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dice_ml
from dice_ml import Dice
import warnings
warnings.filterwarnings('ignore')

BG, FG, SG = '#1D1D20', '#fbfbff', '#909094'
C1, C2, C3, C4 = '#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'axes.edgecolor': SG, 'axes.labelcolor': FG,
                     'xtick.color': FG, 'ytick.color': FG,
                     'text.color': FG, 'grid.color': '#333337', 'grid.alpha': 0.4})

# ─────────────────────────────────────────────
# COMPAS COUNTERFACTUALS
# ─────────────────────────────────────────────
# Rebuild COMPAS as DataFrame for DiCE
_compas_train_cf = pd.DataFrame(X_compas_train, columns=compas_feature_cols)
_compas_train_cf['two_year_recid'] = y_compas_train
_compas_test_cf  = pd.DataFrame(X_compas_test,  columns=compas_feature_cols)
_compas_test_cf['two_year_recid'] = y_compas_test

_d_compas = dice_ml.Data(
    dataframe=_compas_train_cf,
    continuous_features=compas_feature_cols,
    outcome_name='two_year_recid'
)
_m_compas = dice_ml.Model(model=best_compas_model, backend='sklearn')
_exp_compas = Dice(_d_compas, _m_compas, method='random')

# Select 3 instances predicted as recidivist (unfavorable)
_pred_compas = best_compas_model.predict(X_compas_test)
_unfav_idx = np.where(_pred_compas == 1)[0][:3]

print("=" * 70)
print("PHASE 6: COUNTERFACTUAL EXPLANATIONS — COMPAS")
print("=" * 70)

compas_cf_rows = []
for _idx in _unfav_idx:
    _query = _compas_test_cf[compas_feature_cols].iloc[[_idx]]
    _cf = _exp_compas.generate_counterfactuals(
        _query, total_CFs=3, desired_class='opposite',
        features_to_vary=compas_feature_cols)
    _cf_df = _cf.cf_examples_list[0].final_cfs_df
    print(f"\n  Instance {_idx} (Original — Predicted Recidivist):")
    print(f"  {_query.round(3).to_string(index=False)}")
    print(f"\n  Counterfactuals (changes needed to flip prediction):")
    print(_cf_df[compas_feature_cols].round(3).to_string(index=False))
    compas_cf_rows.append({
        'instance': int(_idx),
        'original': _query.values[0].tolist(),
        'counterfactuals': _cf_df[compas_feature_cols].values.tolist()
    })

# ─────────────────────────────────────────────
# ADULT COUNTERFACTUALS
# ─────────────────────────────────────────────
_adult_num_feat = adult_num_cols
_adult_train_cf = pd.DataFrame(
    X_adult_train[:, :len(adult_num_cols)], columns=adult_num_cols)
_adult_train_cf['income'] = y_adult_train

_adult_test_cf = pd.DataFrame(
    X_adult_test[:, :len(adult_num_cols)], columns=adult_num_cols)
_adult_test_cf['income'] = y_adult_test

_d_adult = dice_ml.Data(
    dataframe=_adult_train_cf,
    continuous_features=adult_num_cols,
    outcome_name='income'
)

# Use a simple sklearn model on numeric-only for DiCE
from sklearn.linear_model import LogisticRegression as LR
_lr_num = LR(max_iter=500, random_state=42)
_lr_num.fit(X_adult_train[:, :len(adult_num_cols)], y_adult_train)

_m_adult = dice_ml.Model(model=_lr_num, backend='sklearn')
_exp_adult = Dice(_d_adult, _m_adult, method='random')

_pred_adult_num = _lr_num.predict(X_adult_test[:, :len(adult_num_cols)])
_unfav_adult = np.where(_pred_adult_num == 0)[0][:3]

print("\n" + "=" * 70)
print("PHASE 6: COUNTERFACTUAL EXPLANATIONS — ADULT INCOME")
print("=" * 70)

adult_cf_rows = []
for _idx in _unfav_adult:
    _query = _adult_test_cf[adult_num_cols].iloc[[_idx]]
    _cf = _exp_adult.generate_counterfactuals(
        _query, total_CFs=3, desired_class='opposite',
        features_to_vary=adult_num_cols)
    _cf_df = _cf.cf_examples_list[0].final_cfs_df
    print(f"\n  Instance {_idx} (Original — Predicted <=50K):")
    print(f"  {_query.round(2).to_string(index=False)}")
    print(f"\n  Counterfactuals (changes needed to get >50K):")
    print(_cf_df[adult_num_cols].round(2).to_string(index=False))
    adult_cf_rows.append({
        'instance': int(_idx),
        'original': _query.values[0].tolist(),
        'counterfactuals': _cf_df[adult_num_cols].values.tolist()
    })

# ─────────────────────────────────────────────
# VISUALIZATION: COMPAS CF COMPARISON
# ─────────────────────────────────────────────
_idx0 = _unfav_idx[0]
_query0 = _compas_test_cf[compas_feature_cols].iloc[[_idx0]]
_cf0 = _exp_compas.generate_counterfactuals(
    _query0, total_CFs=2, desired_class='opposite',
    features_to_vary=compas_feature_cols)
_cf0_df = _cf0.cf_examples_list[0].final_cfs_df[compas_feature_cols]

cf_comparison_fig, ax = plt.subplots(figsize=(10, 4))
_x = np.arange(len(compas_feature_cols))
_w = 0.25
ax.bar(_x - _w, _query0.values[0], width=_w, color=C4, label='Original (Recidivist)', edgecolor='none')
for _ci, (_, _cfrow) in enumerate(_cf0_df.iterrows()):
    ax.bar(_x + _ci * _w, _cfrow.values, width=_w,
           color=[C1, C2][_ci], label=f'Counterfactual {_ci+1}', edgecolor='none', alpha=0.85)
ax.set_xticks(_x)
ax.set_xticklabels(compas_feature_cols, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Feature Value (Scaled)', color=FG)
ax.set_title('COMPAS: Original vs Counterfactual Feature Values',
             color=FG, fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
cf_comparison_fig.tight_layout()

import json
with open('compas_counterfactuals.json', 'w') as _f:
    json.dump(compas_cf_rows, _f, indent=2)
with open('adult_counterfactuals.json', 'w') as _f:
    json.dump(adult_cf_rows, _f, indent=2)

print("\n✅ Saved: compas_counterfactuals.json, adult_counterfactuals.json")
print("✅ PHASE 6 COMPLETE — Counterfactual explanations generated.")
print("\nNote: Sensitive attributes (race, sex) are IMMUTABLE by design.")
print("Only actionable features varied in counterfactuals.")
