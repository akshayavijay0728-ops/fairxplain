"""
Name: Phase 4: Fairness Audit
Description: Computes fairness metrics (demographic parity, equalized odds, equal opportunity) across COMPAS and Adult datasets for race/sex groups. Generates a bias scorecard table and heatmap visualization, saves results to CSV and JSON.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                                equalized_odds_difference,
                                demographic_parity_ratio)
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

BG, FG, SG = '#1D1D20', '#fbfbff', '#909094'
C1, C2, C3, C4 = '#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'axes.edgecolor': SG, 'axes.labelcolor': FG,
                     'xtick.color': FG, 'ytick.color': FG,
                     'text.color': FG, 'grid.color': '#333337', 'grid.alpha': 0.4})

def compute_fairness_metrics(y_true, y_pred, y_prob, sensitive_attr, dataset_name):
    results = {}
    _dp_diff  = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_attr)
    _dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_attr)
    _eo_diff  = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_attr)

    # Equal Opportunity (TPR difference)
    _groups = np.unique(sensitive_attr)
    _tpr = {}
    _fpr = {}
    _ppv = {}
    for g in _groups:
        _mask = sensitive_attr == g
        _yt = y_true[_mask]; _yp = y_pred[_mask]
        _tp = np.sum((_yt == 1) & (_yp == 1))
        _fn = np.sum((_yt == 1) & (_yp == 0))
        _fp = np.sum((_yt == 0) & (_yp == 1))
        _tn = np.sum((_yt == 0) & (_yp == 0))
        _tpr[g] = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
        _fpr[g] = _fp / (_fp + _tn) if (_fp + _tn) > 0 else 0
        _ppv[g] = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0

    _tpr_vals = list(_tpr.values())
    _eo_diff_manual = max(_tpr_vals) - min(_tpr_vals)

    _mf = MetricFrame(
        metrics={'accuracy': accuracy_score,
                 'precision': lambda yt, yp: precision_score(yt, yp, zero_division=0),
                 'recall': lambda yt, yp: recall_score(yt, yp, zero_division=0)},
        y_true=y_true, y_pred=y_pred,
        sensitive_features=sensitive_attr
    )

    results = {
        'demographic_parity_difference': round(float(_dp_diff), 4),
        'demographic_parity_ratio': round(float(_dp_ratio), 4),
        'equalized_odds_difference': round(float(_eo_diff), 4),
        'equal_opportunity_difference': round(float(_eo_diff_manual), 4),
        'tpr_by_group': {str(k): round(float(v), 4) for k, v in _tpr.items()},
        'fpr_by_group': {str(k): round(float(v), 4) for k, v in _fpr.items()},
        'ppv_by_group': {str(k): round(float(v), 4) for k, v in _ppv.items()},
    }
    return results, _mf

# ─────────────────────────────────────────────
# COMPAS — RACE
# ─────────────────────────────────────────────
_y_pred_compas = best_compas_model.predict(X_compas_test)
_y_prob_compas = best_compas_model.predict_proba(X_compas_test)[:, 1]

# Focus on African-American vs Caucasian (main fairness concern)
_race_mask = np.isin(race_compas_test, ['African-American', 'Caucasian'])
_y_true_c  = y_compas_test[_race_mask]
_y_pred_c  = _y_pred_compas[_race_mask]
_y_prob_c  = _y_prob_compas[_race_mask]
_race_c    = race_compas_test[_race_mask]

compas_race_metrics, compas_race_mf = compute_fairness_metrics(
    _y_true_c, _y_pred_c, _y_prob_c, _race_c, 'COMPAS-Race')

# COMPAS — SEX
compas_sex_metrics, compas_sex_mf = compute_fairness_metrics(
    y_compas_test, _y_pred_compas, _y_prob_compas, sex_compas_test, 'COMPAS-Sex')

# ─────────────────────────────────────────────
# ADULT — SEX
# ─────────────────────────────────────────────
_y_pred_adult = best_adult_model.predict(X_adult_test)
_y_prob_adult = best_adult_model.predict_proba(X_adult_test)[:, 1]

adult_sex_metrics, adult_sex_mf = compute_fairness_metrics(
    y_adult_test, _y_pred_adult, _y_prob_adult, sex_adult_test, 'Adult-Sex')

# ADULT — RACE (White vs Black)
_race_mask_a = np.isin(race_adult_test, ['White', 'Black'])
_y_true_a  = y_adult_test[_race_mask_a]
_y_pred_a  = _y_pred_adult[_race_mask_a]
_y_prob_a  = _y_prob_adult[_race_mask_a]
_race_a    = race_adult_test[_race_mask_a]

adult_race_metrics, adult_race_mf = compute_fairness_metrics(
    _y_true_a, _y_pred_a, _y_prob_a, _race_a, 'Adult-Race')

# ─────────────────────────────────────────────
# FAIRNESS SCORECARD TABLE
# ─────────────────────────────────────────────
fairness_scorecard = pd.DataFrame({
    'Metric': ['Demographic Parity Diff', 'Demographic Parity Ratio',
               'Equalized Odds Diff', 'Equal Opportunity Diff'],
    'COMPAS (Race)': [
        compas_race_metrics['demographic_parity_difference'],
        compas_race_metrics['demographic_parity_ratio'],
        compas_race_metrics['equalized_odds_difference'],
        compas_race_metrics['equal_opportunity_difference']],
    'COMPAS (Sex)': [
        compas_sex_metrics['demographic_parity_difference'],
        compas_sex_metrics['demographic_parity_ratio'],
        compas_sex_metrics['equalized_odds_difference'],
        compas_sex_metrics['equal_opportunity_difference']],
    'Adult (Sex)': [
        adult_sex_metrics['demographic_parity_difference'],
        adult_sex_metrics['demographic_parity_ratio'],
        adult_sex_metrics['equalized_odds_difference'],
        adult_sex_metrics['equal_opportunity_difference']],
    'Adult (Race)': [
        adult_race_metrics['demographic_parity_difference'],
        adult_race_metrics['demographic_parity_ratio'],
        adult_race_metrics['equalized_odds_difference'],
        adult_race_metrics['equal_opportunity_difference']],
})

print("=" * 80)
print("FAIRXPLAIN — FAIRNESS AUDIT SCORECARD")
print("=" * 80)
print(fairness_scorecard.to_string(index=False))

print("\n📊 COMPAS TPR by Race (African-American vs Caucasian):")
for g, v in compas_race_metrics['tpr_by_group'].items():
    print(f"  {g:<25} TPR={v:.3f}")

print("\n📊 Adult Income TPR by Sex:")
for g, v in adult_sex_metrics['tpr_by_group'].items():
    print(f"  {g:<25} TPR={v:.3f}")

# ─────────────────────────────────────────────
# BIAS HEATMAP
# ─────────────────────────────────────────────
_heatmap_data = np.array([
    [compas_race_metrics['demographic_parity_difference'],
     compas_sex_metrics['demographic_parity_difference'],
     adult_sex_metrics['demographic_parity_difference'],
     adult_race_metrics['demographic_parity_difference']],
    [1 - compas_race_metrics['demographic_parity_ratio'],
     1 - compas_sex_metrics['demographic_parity_ratio'],
     1 - adult_sex_metrics['demographic_parity_ratio'],
     1 - adult_race_metrics['demographic_parity_ratio']],
    [compas_race_metrics['equalized_odds_difference'],
     compas_sex_metrics['equalized_odds_difference'],
     adult_sex_metrics['equalized_odds_difference'],
     adult_race_metrics['equalized_odds_difference']],
    [compas_race_metrics['equal_opportunity_difference'],
     compas_sex_metrics['equal_opportunity_difference'],
     adult_sex_metrics['equal_opportunity_difference'],
     adult_race_metrics['equal_opportunity_difference']],
])

bias_heatmap_fig, ax = plt.subplots(figsize=(10, 5))
_im = ax.imshow(np.abs(_heatmap_data), cmap='YlOrRd', vmin=0, vmax=0.35, aspect='auto')
plt.colorbar(_im, ax=ax, label='Bias Magnitude')
_rows = ['DP Diff', 'DP Ratio Gap', 'EO Diff', 'Eq. Opportunity Diff']
_cols = ['COMPAS\n(Race)', 'COMPAS\n(Sex)', 'Adult\n(Sex)', 'Adult\n(Race)']
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(_cols, fontsize=10); ax.set_yticklabels(_rows, fontsize=10)
for ri in range(4):
    for ci in range(4):
        ax.text(ci, ri, f'{_heatmap_data[ri, ci]:.3f}',
                ha='center', va='center', fontsize=10,
                color='black' if abs(_heatmap_data[ri, ci]) < 0.2 else 'white',
                fontweight='bold')
ax.set_title('Fairness Bias Heatmap — All Datasets & Groups',
             color=FG, fontsize=13, fontweight='bold', pad=12)
bias_heatmap_fig.tight_layout()

# Save
fairness_scorecard.to_csv('fairness_scorecard.csv', index=False)
_all_metrics = {
    'compas_race': compas_race_metrics,
    'compas_sex': compas_sex_metrics,
    'adult_sex': adult_sex_metrics,
    'adult_race': adult_race_metrics
}
with open('fairness_audit_results.json', 'w') as _f:
    json.dump(_all_metrics, _f, indent=2)

print("\n✅ Saved: fairness_scorecard.csv, fairness_audit_results.json")
print("✅ PHASE 4 COMPLETE — Fairness audit done.")
