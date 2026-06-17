"""
Name: Phase 3B: LIME Explanations
Description: Generates LIME (Local Interpretable Model-agnostic Explanations) for individual predictions on COMPAS and Adult Income datasets, comparing top contributing features with SHAP explanations. Creates 3 visualizations per dataset showing positive/negative feature impacts for three case types (predicted positive, predicted negative, borderline).
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

BG, FG, SG = '#1D1D20', '#fbfbff', '#909094'
C1, C2, C3, C4 = '#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'axes.edgecolor': SG, 'axes.labelcolor': FG,
                     'xtick.color': FG, 'ytick.color': FG,
                     'text.color': FG, 'grid.color': '#333337', 'grid.alpha': 0.4})

# ─────────────────────────────────────────────
# COMPAS LIME
# ─────────────────────────────────────────────
_compas_lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_compas_train,
    feature_names=compas_feature_cols,
    class_names=['No Recidivism', 'Recidivism'],
    mode='classification',
    random_state=42
)

# Pick 3 instances: predicted 1, predicted 0, borderline
_compas_probs_test = best_compas_model.predict_proba(X_compas_test)[:, 1]
_compas_preds_test = best_compas_model.predict(X_compas_test)
_idx_c1 = int(np.where(_compas_preds_test == 1)[0][2])
_idx_c0 = int(np.where(_compas_preds_test == 0)[0][2])
_idx_cb = int(np.argmin(np.abs(_compas_probs_test - 0.5)))

_compas_lime_instances = [
    (_idx_c1, 'Predicted: Recidivism',     C4),
    (_idx_c0, 'Predicted: No Recidivism',  C1),
    (_idx_cb, 'Borderline Case',           C2)
]

lime_compas_fig, axes_c = plt.subplots(1, 3, figsize=(15, 5))
for ax_l, (_idx, _title, _clr) in zip(axes_c, _compas_lime_instances):
    _exp = _compas_lime_exp.explain_instance(
        X_compas_test[_idx], best_compas_model.predict_proba, num_features=5)
    _lime_vals = dict(_exp.as_list())
    _feats = list(_lime_vals.keys())
    _vals  = list(_lime_vals.values())
    _colors = [C4 if v > 0 else C1 for v in _vals]
    ax_l.barh(_feats, _vals, color=_colors, edgecolor='none')
    ax_l.axvline(0, color=SG, linewidth=0.8)
    _prob = _compas_probs_test[_idx]
    ax_l.set_title(f'{_title}\nProb={_prob:.3f}', color=FG, fontsize=9, fontweight='bold')
    ax_l.set_xlabel('LIME Weight', color=FG)
    ax_l.tick_params(labelsize=7)
    ax_l.grid(axis='x', alpha=0.3)
lime_compas_fig.suptitle('LIME Instance Explanations — COMPAS', color=FG, fontsize=13, fontweight='bold')
lime_compas_fig.tight_layout()

# ─────────────────────────────────────────────
# ADULT LIME
# ─────────────────────────────────────────────
_adult_lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_adult_train,
    feature_names=adult_feature_names,
    class_names=['<=50K', '>50K'],
    mode='classification',
    random_state=42
)

_adult_probs_test = best_adult_model.predict_proba(X_adult_test)[:, 1]
_adult_preds_test = best_adult_model.predict(X_adult_test)
_idx_a1 = int(np.where(_adult_preds_test == 1)[0][3])
_idx_a0 = int(np.where(_adult_preds_test == 0)[0][3])
_idx_ab = int(np.argmin(np.abs(_adult_probs_test - 0.5)))

_adult_lime_instances = [
    (_idx_a1, 'Predicted: >50K Income',  C3),
    (_idx_a0, 'Predicted: <=50K Income', C4),
    (_idx_ab, 'Borderline Case',         C2)
]

lime_adult_fig, axes_a = plt.subplots(1, 3, figsize=(16, 5))
for ax_l, (_idx, _title, _clr) in zip(axes_a, _adult_lime_instances):
    _exp = _adult_lime_exp.explain_instance(
        X_adult_test[_idx], best_adult_model.predict_proba, num_features=8)
    _lime_vals = dict(_exp.as_list())
    _feats = [f[:25] for f in _lime_vals.keys()]
    _vals  = list(_lime_vals.values())
    _colors = [C3 if v > 0 else C4 for v in _vals]
    ax_l.barh(_feats, _vals, color=_colors, edgecolor='none')
    ax_l.axvline(0, color=SG, linewidth=0.8)
    _prob = _adult_probs_test[_idx]
    ax_l.set_title(f'{_title}\nProb={_prob:.3f}', color=FG, fontsize=9, fontweight='bold')
    ax_l.set_xlabel('LIME Weight', color=FG)
    ax_l.tick_params(labelsize=7)
    ax_l.grid(axis='x', alpha=0.3)
lime_adult_fig.suptitle('LIME Instance Explanations — Adult Income', color=FG, fontsize=13, fontweight='bold')
lime_adult_fig.tight_layout()

# ─────────────────────────────────────────────
# LIME vs SHAP COMPARISON PRINTOUT
# ─────────────────────────────────────────────
print("=" * 65)
print("LIME vs SHAP COMPARISON — COMPAS")
print("=" * 65)
_exp_c = _compas_lime_exp.explain_instance(
    X_compas_test[_idx_c1], best_compas_model.predict_proba, num_features=5)
_lime_top_c = [f[0] for f in sorted(_exp_c.as_list(), key=lambda x: abs(x[1]), reverse=True)]
_shap_top_c = list(np.array(compas_feature_cols)[np.argsort(np.abs(
    shap.LinearExplainer(best_compas_model, X_compas_train).shap_values(X_compas_test[[_idx_c1]])[0]))[::-1]])
print(f"\n  Instance {_idx_c1} (Predicted: Recidivism)")
print(f"  LIME top features  : {_lime_top_c[:3]}")
print(f"  SHAP top features  : {_shap_top_c[:3]}")

print("\n" + "=" * 65)
print("LIME vs SHAP COMPARISON — ADULT INCOME")
print("=" * 65)
_exp_a = _adult_lime_exp.explain_instance(
    X_adult_test[_idx_a1], best_adult_model.predict_proba, num_features=8)
_lime_top_a = [f[0][:30] for f in sorted(_exp_a.as_list(), key=lambda x: abs(x[1]), reverse=True)[:3]]
_adult_explainer2 = shap.TreeExplainer(best_adult_model)
_shap_top_a = [adult_feature_names[i][:30] for i in np.argsort(
    np.abs(_adult_explainer2.shap_values(X_adult_test[[_idx_a1]])[0]))[::-1][:3]]
print(f"\n  Instance {_idx_a1} (Predicted: >50K)")
print(f"  LIME top features  : {_lime_top_a}")
print(f"  SHAP top features  : {_shap_top_a}")

print("\n✅ PHASE 3B COMPLETE — LIME explanations generated.")
print("✅ PHASE 3 (XAI) FULLY COMPLETE — SHAP + LIME done.")
