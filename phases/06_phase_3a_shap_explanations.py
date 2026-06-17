"""
Name: Phase 3A: SHAP Explanations
Description: Computes SHAP explainability analysis for COMPAS and Adult Income models. Generates global feature importance bar charts and individual instance waterfall plots showing how features drive predictions (recidivism risk, income threshold) using Linear and Tree explainers.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

BG, FG, SG = '#1D1D20', '#fbfbff', '#909094'
C1, C2, C3, C4 = '#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'axes.edgecolor': SG, 'axes.labelcolor': FG,
                     'xtick.color': FG, 'ytick.color': FG,
                     'text.color': FG, 'grid.color': '#333337', 'grid.alpha': 0.4})

# ─────────────────────────────────────────────
# COMPAS SHAP — Logistic Regression → Linear Explainer
# ─────────────────────────────────────────────
print("Computing SHAP for COMPAS (Logistic Regression)...")
_compas_explainer = shap.LinearExplainer(best_compas_model, X_compas_train)
_compas_shap_vals = _compas_explainer.shap_values(X_compas_test)
_compas_feat_arr  = np.array(compas_feature_cols)

# Global bar chart of mean |SHAP|
shap_compas_global_fig, ax = plt.subplots(figsize=(8, 4))
_mean_shap_c = np.abs(_compas_shap_vals).mean(axis=0)
_order_c = np.argsort(_mean_shap_c)
ax.barh(_compas_feat_arr[_order_c], _mean_shap_c[_order_c], color=C1, edgecolor='none')
ax.set_xlabel('Mean |SHAP Value|', color=FG)
ax.set_title('SHAP Global Feature Importance — COMPAS\n(Logistic Regression)',
             color=FG, fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
shap_compas_global_fig.tight_layout()

# Waterfall for 2 instances (one predicted 0, one predicted 1)
_compas_preds = best_compas_model.predict(X_compas_test)
_idx_pos = int(np.where(_compas_preds == 1)[0][0])
_idx_neg = int(np.where(_compas_preds == 0)[0][0])

shap_compas_wf_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for ax_wf, _idx, _label, _color in [(ax1, _idx_pos, 'Predicted: Recidivism', C4),
                                      (ax2, _idx_neg, 'Predicted: No Recidivism', C1)]:
    _sv = _compas_shap_vals[_idx]
    _base = _compas_explainer.expected_value
    _order = np.argsort(np.abs(_sv))[::-1]
    _feats = _compas_feat_arr[_order]
    _vals  = _sv[_order]
    _colors = [C4 if v > 0 else C1 for v in _vals]
    ax_wf.barh(_feats, _vals, color=_colors, edgecolor='none')
    ax_wf.axvline(0, color=SG, linewidth=0.8)
    ax_wf.set_title(f'SHAP Waterfall — {_label}', color=FG, fontsize=10, fontweight='bold')
    ax_wf.set_xlabel('SHAP Value', color=FG)
    ax_wf.grid(axis='x', alpha=0.3)
shap_compas_wf_fig.suptitle('COMPAS: Individual SHAP Explanations', color=FG, fontsize=13, fontweight='bold')
shap_compas_wf_fig.tight_layout()

# ─────────────────────────────────────────────
# ADULT SHAP — XGBoost → TreeExplainer
# ─────────────────────────────────────────────
print("Computing SHAP for Adult Income (XGBoost)...")
_adult_explainer = shap.TreeExplainer(best_adult_model)
# Use a sample of 500 for speed
_adult_sample_idx = np.random.RandomState(42).choice(len(X_adult_test), 500, replace=False)
_X_adult_sample   = X_adult_test[_adult_sample_idx]
_adult_shap_vals  = _adult_explainer.shap_values(_X_adult_sample)
_adult_feat_arr   = np.array(adult_feature_names)

# Top 15 global SHAP
_mean_shap_a = np.abs(_adult_shap_vals).mean(axis=0)
_top15_a = np.argsort(_mean_shap_a)[-15:]

shap_adult_global_fig, ax3 = plt.subplots(figsize=(9, 6))
ax3.barh(_adult_feat_arr[_top15_a], _mean_shap_a[_top15_a], color=C2, edgecolor='none')
ax3.set_xlabel('Mean |SHAP Value|', color=FG)
ax3.set_title('SHAP Top-15 Feature Importance — Adult Income\n(XGBoost)',
              color=FG, fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
shap_adult_global_fig.tight_layout()

# Waterfall for Adult
_adult_preds_s = best_adult_model.predict(_X_adult_sample)
_idx_pos_a = int(np.where(_adult_preds_s == 1)[0][0])
_idx_neg_a = int(np.where(_adult_preds_s == 0)[0][0])

shap_adult_wf_fig, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 6))
for ax_wf, _idx, _label, _color in [(ax4, _idx_pos_a, 'Predicted: >50K Income', C3),
                                      (ax5, _idx_neg_a, 'Predicted: <=50K Income', C4)]:
    _sv  = _adult_shap_vals[_idx]
    _top_idx = np.argsort(np.abs(_sv))[::-1][:15]
    _feats = _adult_feat_arr[_top_idx]
    _vals  = _sv[_top_idx]
    _colors = [C3 if v > 0 else C4 for v in _vals]
    ax_wf.barh(_feats, _vals, color=_colors, edgecolor='none')
    ax_wf.axvline(0, color=SG, linewidth=0.8)
    ax_wf.set_title(f'SHAP Waterfall — {_label}', color=FG, fontsize=9, fontweight='bold')
    ax_wf.set_xlabel('SHAP Value', color=FG)
    ax_wf.tick_params(labelsize=8)
    ax_wf.grid(axis='x', alpha=0.3)
shap_adult_wf_fig.suptitle('Adult Income: Individual SHAP Explanations', color=FG, fontsize=13, fontweight='bold')
shap_adult_wf_fig.tight_layout()

# Summary printout
print("\n" + "=" * 60)
print("SHAP GLOBAL IMPORTANCE — COMPAS")
print("=" * 60)
for feat, imp in zip(_compas_feat_arr[np.argsort(_mean_shap_c)[::-1]], np.sort(_mean_shap_c)[::-1]):
    print(f"  {feat:<30} {imp:.4f}")

print("\n" + "=" * 60)
print("SHAP TOP-10 GLOBAL IMPORTANCE — ADULT INCOME")
print("=" * 60)
for feat, imp in zip(_adult_feat_arr[np.argsort(_mean_shap_a)[::-1][:10]],
                     np.sort(_mean_shap_a)[::-1][:10]):
    print(f"  {feat:<35} {imp:.4f}")

print("\n✅ PHASE 3A COMPLETE — SHAP explanations generated.")
