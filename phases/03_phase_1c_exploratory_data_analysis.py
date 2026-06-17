"""
Name: Phase 1C: Exploratory Data Analysis
Description: Creates 4 exploratory visualizations comparing recidivism & income outcomes by race/sex across COMPAS and Adult datasets, plus statistical summaries of disparities.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Zerve color palette
C1, C2, C3, C4, C5 = '#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF'
BG, FG, SG = '#1D1D20', '#fbfbff', '#909094'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': SG, 'axes.labelcolor': FG,
    'xtick.color': FG, 'ytick.color': FG,
    'text.color': FG, 'grid.color': '#333337',
    'grid.alpha': 0.4, 'font.size': 10
})

# ─────────────────────────────────────────────
# FIGURE 1: COMPAS — Recidivism Rate by Race
# ─────────────────────────────────────────────
_compas_race_rates = (compas_df.groupby('race')['two_year_recid']
                      .agg(['mean', 'count'])
                      .sort_values('mean', ascending=True)
                      .reset_index())

eda_compas_race_fig, ax1 = plt.subplots(figsize=(9, 5))
_colors = [C1, C2, C3, C4, C5, '#ffd400']
_bars = ax1.barh(_compas_race_rates['race'],
                 _compas_race_rates['mean'] * 100,
                 color=_colors[:len(_compas_race_rates)],
                 edgecolor='none', height=0.6)
for bar, (_, row) in zip(_bars, _compas_race_rates.iterrows()):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{row['mean']*100:.1f}%  (n={row['count']:,})",
             va='center', color=FG, fontsize=9)
ax1.set_xlabel('Recidivism Rate (%)', color=FG)
ax1.set_title('COMPAS: 2-Year Recidivism Rate by Race', color=FG,
              fontsize=13, fontweight='bold', pad=12)
ax1.set_xlim(0, 75)
_overall = float(compas_df['two_year_recid'].mean() * 100)
ax1.axvline(_overall, color='#ffd400', linestyle='--',
            linewidth=1.5, label=f"Overall avg: {_overall:.1f}%")
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)
eda_compas_race_fig.tight_layout()

# ─────────────────────────────────────────────
# FIGURE 2: COMPAS — Age Distribution by Outcome
# ─────────────────────────────────────────────
eda_compas_age_fig, ax2 = plt.subplots(figsize=(9, 5))
_rec0 = compas_df[compas_df['two_year_recid'] == 0]['age']
_rec1 = compas_df[compas_df['two_year_recid'] == 1]['age']
ax2.hist(_rec0, bins=30, alpha=0.7, color=C1, label='No Recidivism', edgecolor='none')
ax2.hist(_rec1, bins=30, alpha=0.7, color=C4, label='Recidivism', edgecolor='none')
ax2.set_xlabel('Age', color=FG)
ax2.set_ylabel('Count', color=FG)
ax2.set_title('COMPAS: Age Distribution by Recidivism Outcome',
              color=FG, fontsize=13, fontweight='bold', pad=12)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
eda_compas_age_fig.tight_layout()

# ─────────────────────────────────────────────
# FIGURE 3: Adult — Income Rate by Sex & Race
# ─────────────────────────────────────────────
eda_adult_disparity_fig, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

_sex_rates = adult_df.groupby('sex')['income'].mean() * 100
_sex_labels = list(_sex_rates.index)
_sex_vals = list(_sex_rates.values)
ax3a.bar(_sex_labels, _sex_vals, color=[C1, C2], edgecolor='none', width=0.5)
for idx, val in enumerate(_sex_vals):
    ax3a.text(idx, val + 0.5, f'{val:.1f}%', ha='center',
              color=FG, fontsize=11, fontweight='bold')
ax3a.set_ylabel('Income >50K Rate (%)', color=FG)
ax3a.set_title('Adult: Income Rate by Sex', color=FG, fontsize=12, fontweight='bold')
ax3a.set_ylim(0, 40)
ax3a.grid(axis='y', alpha=0.3)

_race_rates_s = adult_df.groupby('race')['income'].mean().sort_values(ascending=False).head(5)
_race_labels = list(_race_rates_s.index)
_race_vals = list(_race_rates_s.values * 100)
ax3b.bar(range(len(_race_labels)), _race_vals,
         color=[C1, C2, C3, C4, C5], edgecolor='none', width=0.6)
ax3b.set_xticks(range(len(_race_labels)))
ax3b.set_xticklabels([r[:12] for r in _race_labels], rotation=15, ha='right', fontsize=9)
for idx, val in enumerate(_race_vals):
    ax3b.text(idx, val + 0.3, f'{val:.1f}%', ha='center',
              color=FG, fontsize=10, fontweight='bold')
ax3b.set_ylabel('Income >50K Rate (%)', color=FG)
ax3b.set_title('Adult: Income Rate by Race', color=FG, fontsize=12, fontweight='bold')
ax3b.set_ylim(0, 40)
ax3b.grid(axis='y', alpha=0.3)
eda_adult_disparity_fig.suptitle('Adult Income Dataset: Group Disparities',
                                  color=FG, fontsize=14, fontweight='bold')
eda_adult_disparity_fig.tight_layout()

# ─────────────────────────────────────────────
# FIGURE 4: COMPAS Correlation Heatmap
# ─────────────────────────────────────────────
eda_corr_fig, ax4 = plt.subplots(figsize=(7, 5))
_corr_cols = ['age', 'priors_count', 'length_of_stay',
              'days_b_screening_arrest', 'decile_score', 'two_year_recid']
_corr = compas_df[_corr_cols].corr()
_im = ax4.imshow(_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(_im, ax=ax4)
_short = ['Age', 'Priors', 'Stay', 'Days', 'COMPAS\nScore', 'Recidivism']
ax4.set_xticks(range(len(_corr_cols)))
ax4.set_yticks(range(len(_corr_cols)))
ax4.set_xticklabels(_short, rotation=30, ha='right', fontsize=9)
ax4.set_yticklabels(_short, fontsize=9)
for ri in range(len(_corr_cols)):
    for ci in range(len(_corr_cols)):
        ax4.text(ci, ri, f'{_corr.values[ri, ci]:.2f}',
                 ha='center', va='center', fontsize=8,
                 color='white' if abs(_corr.values[ri, ci]) > 0.5 else FG)
ax4.set_title('COMPAS: Feature Correlation Matrix',
              color=FG, fontsize=12, fontweight='bold', pad=10)
eda_corr_fig.tight_layout()

# ─────────────────────────────────────────────
# Print Statistical Summaries
# ─────────────────────────────────────────────
print("=" * 60)
print("COMPAS STATISTICAL SUMMARY")
print("=" * 60)
print(compas_df[['age', 'priors_count', 'length_of_stay', 'decile_score']].describe().round(2).to_string())

print("\n" + "=" * 60)
print("ADULT STATISTICAL SUMMARY")
print("=" * 60)
print(adult_df[['age', 'education_num', 'capital_gain', 'hours_per_week']].describe().round(2).to_string())

print("\n" + "=" * 60)
print("FAIRNESS DISPARITY SUMMARY")
print("=" * 60)
print("\nCOMPAS — Recidivism Rate by Race:")
for _r, _v in compas_df.groupby('race')['two_year_recid'].mean().sort_values(ascending=False).items():
    print(f"  {_r:<25} {_v*100:.1f}%")

print("\nCOMPAS — Recidivism Rate by Sex:")
for _s, _v in compas_df.groupby('sex')['two_year_recid'].mean().items():
    print(f"  {_s:<25} {_v*100:.1f}%")

print("\nAdult — Income >50K Rate by Sex:")
for _s, _v in adult_df.groupby('sex')['income'].mean().items():
    print(f"  {_s:<25} {_v*100:.1f}%")

print("\nAdult — Income >50K Rate by Race:")
for _r, _v in adult_df.groupby('race')['income'].mean().sort_values(ascending=False).items():
    print(f"  {_r:<25} {_v*100:.1f}%")

print("\n✅ PHASE 1C COMPLETE — EDA charts generated.")
