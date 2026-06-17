"""
Name: Phase 5: Bias Mitigation
Description: Applies three bias mitigation strategies (Reweighing, Exponentiated Gradient, Threshold Optimizer) to a COMPAS model, comparing fairness metrics (demographic parity, equalized odds) against accuracy to visualize the fairness-accuracy tradeoff.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
import warnings
warnings.filterwarnings('ignore')

BG, FG, SG = '#1D1D20', '#fbfbff', '#909094'
C1, C2, C3, C4, C5 = '#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF'
RANDOM_STATE = 42

plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'axes.edgecolor': SG, 'axes.labelcolor': FG,
                     'xtick.color': FG, 'ytick.color': FG,
                     'text.color': FG, 'grid.color': '#333337', 'grid.alpha': 0.4})

# ─────────────────────────────────────────────
# BASELINE METRICS (already computed)
# ─────────────────────────────────────────────
_y_pred_base = best_compas_model.predict(X_compas_test)
_base_acc = accuracy_score(y_compas_test, _y_pred_base)
_base_dp  = demographic_parity_difference(y_compas_test, _y_pred_base,
                                           sensitive_features=race_compas_test)
_base_eo  = equalized_odds_difference(y_compas_test, _y_pred_base,
                                       sensitive_features=race_compas_test)
_base_f1  = f1_score(y_compas_test, _y_pred_base, zero_division=0)

print("=" * 65)
print("PHASE 5: BIAS MITIGATION — COMPAS DATASET")
print("=" * 65)
print(f"\nBaseline (Logistic Regression):")
print(f"  Accuracy={_base_acc:.4f} | F1={_base_f1:.4f} | "
      f"DP-Diff={_base_dp:.4f} | EO-Diff={_base_eo:.4f}")

# ─────────────────────────────────────────────
# STRATEGY 1: REWEIGHING (AIF360 Pre-processing)
# ─────────────────────────────────────────────
print("\n--- Strategy 1: Reweighing (Pre-processing) ---")
_compas_train_df = pd.DataFrame(X_compas_train,
    columns=[f'f{i}' for i in range(X_compas_train.shape[1])])
_compas_train_df['label'] = y_compas_train
_compas_train_df['race_enc'] = (race_compas_train == 'African-American').astype(int)

_compas_test_df = pd.DataFrame(X_compas_test,
    columns=[f'f{i}' for i in range(X_compas_test.shape[1])])
_compas_test_df['label'] = y_compas_test
_compas_test_df['race_enc'] = (race_compas_test == 'African-American').astype(int)

_feat_cols = [f'f{i}' for i in range(X_compas_train.shape[1])]
_aif_train = BinaryLabelDataset(df=_compas_train_df,
    label_names=['label'], protected_attribute_names=['race_enc'],
    favorable_label=0, unfavorable_label=1)
_aif_test  = BinaryLabelDataset(df=_compas_test_df,
    label_names=['label'], protected_attribute_names=['race_enc'],
    favorable_label=0, unfavorable_label=1)

_rw = Reweighing(unprivileged_groups=[{'race_enc': 1}],
                 privileged_groups=[{'race_enc': 0}])
_aif_train_rw = _rw.fit_transform(_aif_train)
_sample_weights = _aif_train_rw.instance_weights

_lr_rw = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
_lr_rw.fit(X_compas_train, y_compas_train, sample_weight=_sample_weights)
_y_pred_rw = _lr_rw.predict(X_compas_test)
_rw_acc = accuracy_score(y_compas_test, _y_pred_rw)
_rw_dp  = demographic_parity_difference(y_compas_test, _y_pred_rw,
                                         sensitive_features=race_compas_test)
_rw_eo  = equalized_odds_difference(y_compas_test, _y_pred_rw,
                                     sensitive_features=race_compas_test)
_rw_f1  = f1_score(y_compas_test, _y_pred_rw, zero_division=0)
print(f"  Accuracy={_rw_acc:.4f} | F1={_rw_f1:.4f} | "
      f"DP-Diff={_rw_dp:.4f} | EO-Diff={_rw_eo:.4f}")

# ─────────────────────────────────────────────
# STRATEGY 2: EXPONENTIATED GRADIENT (In-processing)
# ─────────────────────────────────────────────
print("\n--- Strategy 2: Exponentiated Gradient — EqualizedOdds (In-processing) ---")
_race_binary_train = (race_compas_train == 'African-American').astype(int)
_race_binary_test  = (race_compas_test  == 'African-American').astype(int)

_lr_base = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
_eg = ExponentiatedGradient(_lr_base, constraints=EqualizedOdds(), eps=0.05)
_eg.fit(X_compas_train, y_compas_train, sensitive_features=_race_binary_train)
_y_pred_eg = _eg.predict(X_compas_test)
_eg_acc = accuracy_score(y_compas_test, _y_pred_eg)
_eg_dp  = demographic_parity_difference(y_compas_test, _y_pred_eg,
                                         sensitive_features=race_compas_test)
_eg_eo  = equalized_odds_difference(y_compas_test, _y_pred_eg,
                                     sensitive_features=race_compas_test)
_eg_f1  = f1_score(y_compas_test, _y_pred_eg, zero_division=0)
print(f"  Accuracy={_eg_acc:.4f} | F1={_eg_f1:.4f} | "
      f"DP-Diff={_eg_dp:.4f} | EO-Diff={_eg_eo:.4f}")

# ─────────────────────────────────────────────
# STRATEGY 3: THRESHOLD OPTIMIZER (Post-processing)
# ─────────────────────────────────────────────
print("\n--- Strategy 3: ThresholdOptimizer (Post-processing) ---")
_to = ThresholdOptimizer(
    estimator=best_compas_model,
    constraints='equalized_odds',
    objective='accuracy_score',
    predict_method='predict_proba'
)
_to.fit(X_compas_train, y_compas_train, sensitive_features=race_compas_train)
_y_pred_to = _to.predict(X_compas_test, sensitive_features=race_compas_test)
_to_acc = accuracy_score(y_compas_test, _y_pred_to)
_to_dp  = demographic_parity_difference(y_compas_test, _y_pred_to,
                                         sensitive_features=race_compas_test)
_to_eo  = equalized_odds_difference(y_compas_test, _y_pred_to,
                                     sensitive_features=race_compas_test)
_to_f1  = f1_score(y_compas_test, _y_pred_to, zero_division=0)
print(f"  Accuracy={_to_acc:.4f} | F1={_to_f1:.4f} | "
      f"DP-Diff={_to_dp:.4f} | EO-Diff={_to_eo:.4f}")

# ─────────────────────────────────────────────
# CONSOLIDATED TRADEOFF TABLE
# ─────────────────────────────────────────────
mitigation_tradeoff = pd.DataFrame({
    'Strategy':  ['Baseline', 'Reweighing', 'Exponentiated Gradient', 'Threshold Optimizer'],
    'Accuracy':  [round(_base_acc, 4), round(_rw_acc, 4), round(_eg_acc, 4), round(_to_acc, 4)],
    'F1':        [round(_base_f1, 4), round(_rw_f1, 4), round(_eg_f1, 4), round(_to_f1, 4)],
    'DP_Diff':   [round(_base_dp, 4), round(_rw_dp, 4), round(_eg_dp, 4), round(_to_dp, 4)],
    'EO_Diff':   [round(_base_eo, 4), round(_rw_eo, 4), round(_eg_eo, 4), round(_to_eo, 4)],
})
print("\n" + "=" * 65)
print("FAIRNESS-ACCURACY TRADEOFF TABLE")
print("=" * 65)
print(mitigation_tradeoff.to_string(index=False))

# ─────────────────────────────────────────────
# TRADEOFF PLOT
# ─────────────────────────────────────────────
tradeoff_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
_colors = [C4, C1, C2, C3]
_markers = ['o', 's', '^', 'D']
_strategies = mitigation_tradeoff['Strategy'].tolist()

for idx, (row, clr, mrk) in enumerate(zip(mitigation_tradeoff.itertuples(), _colors, _markers)):
    ax1.scatter(abs(row.DP_Diff), row.Accuracy,
                color=clr, s=180, marker=mrk, zorder=5, label=row.Strategy)
    ax1.annotate(row.Strategy, (abs(row.DP_Diff), row.Accuracy),
                 textcoords='offset points', xytext=(5, 5), fontsize=8, color=FG)
ax1.set_xlabel('|Demographic Parity Diff| (lower = fairer)', color=FG)
ax1.set_ylabel('Accuracy', color=FG)
ax1.set_title('Fairness-Accuracy Tradeoff\n(COMPAS — DP Diff)', color=FG, fontsize=11, fontweight='bold')
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

for idx, (row, clr, mrk) in enumerate(zip(mitigation_tradeoff.itertuples(), _colors, _markers)):
    ax2.scatter(abs(row.EO_Diff), row.Accuracy,
                color=clr, s=180, marker=mrk, zorder=5, label=row.Strategy)
    ax2.annotate(row.Strategy, (abs(row.EO_Diff), row.Accuracy),
                 textcoords='offset points', xytext=(5, 5), fontsize=8, color=FG)
ax2.set_xlabel('|Equalized Odds Diff| (lower = fairer)', color=FG)
ax2.set_ylabel('Accuracy', color=FG)
ax2.set_title('Fairness-Accuracy Tradeoff\n(COMPAS — EO Diff)', color=FG, fontsize=11, fontweight='bold')
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
tradeoff_fig.suptitle('Bias Mitigation Fairness-Accuracy Tradeoff', color=FG, fontsize=13, fontweight='bold')
tradeoff_fig.tight_layout()

mitigation_tradeoff.to_csv('mitigation_tradeoff.csv', index=False)
print("\n✅ Saved: mitigation_tradeoff.csv")
print("✅ PHASE 5 COMPLETE — Bias mitigation done.")
