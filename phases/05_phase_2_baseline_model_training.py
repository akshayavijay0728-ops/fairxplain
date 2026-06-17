"""
Name: Phase 2: Baseline Model Training
Description: Trains and evaluates three classification models (Logistic Regression, Random Forest, XGBoost) on COMPAS recidivism and Adult income datasets. Compares performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC), selects the best model by AUC for each dataset, and generates ROC curves and feature importance visualizations. Saves trained models and metrics tables.
"""


import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

BG, FG, SG = '#1D1D20', '#fbfbff', '#909094'
C1, C2, C3 = '#A1C9F4', '#FFB482', '#8DE5A1'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'axes.edgecolor': SG, 'axes.labelcolor': FG,
                     'xtick.color': FG, 'ytick.color': FG,
                     'text.color': FG, 'grid.color': '#333337', 'grid.alpha': 0.4})

RANDOM_STATE = 42

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced'),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                         eval_metric='logloss', verbosity=0,
                                         scale_pos_weight=1)
}

def evaluate_model(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    return {
        'Accuracy':  round(accuracy_score(y_te, y_pred), 4),
        'Precision': round(precision_score(y_te, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_te, y_pred, zero_division=0), 4),
        'F1':        round(f1_score(y_te, y_pred, zero_division=0), 4),
        'ROC-AUC':   round(roc_auc_score(y_te, y_prob), 4)
    }, y_prob

# ─────────────────────────────────────────────
# TRAIN & EVALUATE ON BOTH DATASETS
# ─────────────────────────────────────────────
compas_results, adult_results = {}, {}
compas_probs, adult_probs = {}, {}
trained_compas, trained_adult = {}, {}

print("=" * 65)
print("TRAINING MODELS ON COMPAS RECIDIVISM")
print("=" * 65)
for _name, _model in _models.items():
    _metrics, _prob = evaluate_model(
        _model, X_compas_train, y_compas_train, X_compas_test, y_compas_test)
    compas_results[_name] = _metrics
    compas_probs[_name]   = _prob
    trained_compas[_name] = _model
    print(f"  {_name:<22} | Acc={_metrics['Accuracy']:.3f} | "
          f"F1={_metrics['F1']:.3f} | AUC={_metrics['ROC-AUC']:.3f}")

print("\n" + "=" * 65)
print("TRAINING MODELS ON ADULT INCOME")
print("=" * 65)
for _name, _model in _models.items():
    _m2 = type(_model)(**_model.get_params())
    _metrics2, _prob2 = evaluate_model(
        _m2, X_adult_train, y_adult_train, X_adult_test, y_adult_test)
    adult_results[_name] = _metrics2
    adult_probs[_name]   = _prob2
    trained_adult[_name] = _m2
    print(f"  {_name:<22} | Acc={_metrics2['Accuracy']:.3f} | "
          f"F1={_metrics2['F1']:.3f} | AUC={_metrics2['ROC-AUC']:.3f}")

# ─────────────────────────────────────────────
# COMPARISON TABLES
# ─────────────────────────────────────────────
compas_table = pd.DataFrame(compas_results).T.reset_index().rename(columns={'index': 'Model'})
adult_table  = pd.DataFrame(adult_results).T.reset_index().rename(columns={'index': 'Model'})

print("\n" + "=" * 65)
print("COMPAS — FULL MODEL COMPARISON")
print("=" * 65)
print(compas_table.to_string(index=False))

print("\n" + "=" * 65)
print("ADULT INCOME — FULL MODEL COMPARISON")
print("=" * 65)
print(adult_table.to_string(index=False))

# ─────────────────────────────────────────────
# SELECT BEST MODEL (by ROC-AUC)
# ─────────────────────────────────────────────
best_compas_name = compas_table.loc[compas_table['ROC-AUC'].idxmax(), 'Model']
best_adult_name  = adult_table.loc[adult_table['ROC-AUC'].idxmax(), 'Model']
best_compas_model = trained_compas[best_compas_name]
best_adult_model  = trained_adult[best_adult_name]

print(f"\n🏆 Best COMPAS model : {best_compas_name}")
print(f"🏆 Best Adult  model : {best_adult_name}")

# ─────────────────────────────────────────────
# ROC CURVES
# ─────────────────────────────────────────────
roc_curves_fig, (ax_c, ax_a) = plt.subplots(1, 2, figsize=(13, 5))
_palette = [C1, C2, C3]

for idx, (_name, _prob) in enumerate(compas_probs.items()):
    _fpr, _tpr, _ = roc_curve(y_compas_test, _prob)
    ax_c.plot(_fpr, _tpr, color=_palette[idx], linewidth=2,
              label=f"{_name} (AUC={compas_results[_name]['ROC-AUC']:.3f})")
ax_c.plot([0,1],[0,1], '--', color=SG, linewidth=1)
ax_c.set_xlabel('False Positive Rate'); ax_c.set_ylabel('True Positive Rate')
ax_c.set_title('ROC Curves — COMPAS', fontsize=12, fontweight='bold', color=FG)
ax_c.legend(fontsize=9); ax_c.grid(alpha=0.3)

for idx, (_name, _prob) in enumerate(adult_probs.items()):
    _fpr, _tpr, _ = roc_curve(y_adult_test, _prob)
    ax_a.plot(_fpr, _tpr, color=_palette[idx], linewidth=2,
              label=f"{_name} (AUC={adult_results[_name]['ROC-AUC']:.3f})")
ax_a.plot([0,1],[0,1], '--', color=SG, linewidth=1)
ax_a.set_xlabel('False Positive Rate'); ax_a.set_ylabel('True Positive Rate')
ax_a.set_title('ROC Curves — Adult Income', fontsize=12, fontweight='bold', color=FG)
ax_a.legend(fontsize=9); ax_a.grid(alpha=0.3)
roc_curves_fig.tight_layout()

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE (best models)
# ─────────────────────────────────────────────
feature_importance_fig, (ax_fi1, ax_fi2) = plt.subplots(1, 2, figsize=(14, 5))

# COMPAS feature importance
if hasattr(best_compas_model, 'feature_importances_'):
    _imp_c = best_compas_model.feature_importances_
else:
    _imp_c = np.abs(best_compas_model.coef_[0])
_imp_c_sorted = np.argsort(_imp_c)
_feat_c = np.array(compas_feature_cols)
ax_fi1.barh(_feat_c[_imp_c_sorted], _imp_c[_imp_c_sorted], color=C1, edgecolor='none')
ax_fi1.set_title(f'Feature Importance — COMPAS\n({best_compas_name})',
                  color=FG, fontsize=11, fontweight='bold')
ax_fi1.set_xlabel('Importance', color=FG); ax_fi1.grid(axis='x', alpha=0.3)

# Adult top-20 features
if hasattr(best_adult_model, 'feature_importances_'):
    _imp_a = best_adult_model.feature_importances_
else:
    _imp_a = np.abs(best_adult_model.coef_[0])
_top20 = np.argsort(_imp_a)[-20:]
_feat_a = np.array(adult_feature_names)
ax_fi2.barh(_feat_a[_top20], _imp_a[_top20], color=C2, edgecolor='none')
ax_fi2.set_title(f'Top-20 Features — Adult Income\n({best_adult_name})',
                  color=FG, fontsize=11, fontweight='bold')
ax_fi2.set_xlabel('Importance', color=FG); ax_fi2.grid(axis='x', alpha=0.3)
feature_importance_fig.tight_layout()

# Save metrics and models
compas_table.to_csv('compas_model_metrics.csv', index=False)
adult_table.to_csv('adult_model_metrics.csv', index=False)
with open('best_compas_model.pkl', 'wb') as _f: pickle.dump(best_compas_model, _f)
with open('best_adult_model.pkl', 'wb') as _f:  pickle.dump(best_adult_model, _f)
_best = {'compas': best_compas_name, 'adult': best_adult_name}
with open('best_models.json', 'w') as _f: json.dump(_best, _f, indent=2)

print("\n✅ Saved: compas_model_metrics.csv, adult_model_metrics.csv")
print("✅ Saved: best_compas_model.pkl, best_adult_model.pkl")
print("✅ PHASE 2 COMPLETE — Baseline models trained and evaluated.")
