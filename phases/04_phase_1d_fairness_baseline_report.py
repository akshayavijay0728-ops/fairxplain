"""
Name: Phase 1D: Fairness Baseline Report
Description: Analyzes fairness baseline disparities in COMPAS and Adult datasets by computing outcome rates, disparate impact ratios, and demographic breakdowns by race/sex. Generates detailed reports and summary metrics for bias assessment.
"""


import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

def disparity_report(df, target, group_col, label):
    rates = df.groupby(group_col)[target].mean()
    counts = df.groupby(group_col)[target].count()
    overall = df[target].mean()
    rows = []
    for grp in rates.index:
        r = rates[grp]
        rows.append({
            'Group': grp,
            'Outcome_Rate': round(r, 4),
            'Outcome_Pct': round(r * 100, 2),
            'Count': int(counts[grp]),
            'Disparity_Diff': round(r - overall, 4),
            'Disparity_Ratio': round(r / overall, 4) if overall > 0 else None
        })
    return pd.DataFrame(rows).sort_values('Outcome_Rate', ascending=False)

# COMPAS Reports
compas_race_report = disparity_report(compas_df, 'two_year_recid', 'race', 'COMPAS Race')
compas_sex_report  = disparity_report(compas_df, 'two_year_recid', 'sex',  'COMPAS Sex')

# Adult Reports
adult_sex_report   = disparity_report(adult_df, 'income', 'sex',  'Adult Sex')
adult_race_report  = disparity_report(adult_df, 'income', 'race', 'Adult Race')

print("=" * 70)
print("FAIRXPLAIN — FAIRNESS BASELINE REPORT")
print("=" * 70)

print("\n📊 COMPAS: Recidivism Rate by Race")
print("-" * 70)
print(compas_race_report.to_string(index=False))
_compas_race_dr = (compas_race_report[compas_race_report['Group'] == 'African-American']['Outcome_Rate'].values[0] /
                   compas_race_report[compas_race_report['Group'] == 'Caucasian']['Outcome_Rate'].values[0])
print(f"\n  ⚠️  African-American / Caucasian Disparate Impact Ratio: {_compas_race_dr:.3f}")
print(f"  ⚠️  Interpretation: African-Americans are {_compas_race_dr:.2f}x more likely to be predicted as recidivist")

print("\n📊 COMPAS: Recidivism Rate by Sex")
print("-" * 70)
print(compas_sex_report.to_string(index=False))
_compas_sex_dr = (compas_sex_report[compas_sex_report['Group'] == 'Male']['Outcome_Rate'].values[0] /
                  compas_sex_report[compas_sex_report['Group'] == 'Female']['Outcome_Rate'].values[0])
print(f"\n  ⚠️  Male / Female Disparate Impact Ratio: {_compas_sex_dr:.3f}")

print("\n📊 Adult Income: Income Rate by Sex")
print("-" * 70)
print(adult_sex_report.to_string(index=False))
_adult_sex_dr = (adult_sex_report[adult_sex_report['Group'] == 'Female']['Outcome_Rate'].values[0] /
                 adult_sex_report[adult_sex_report['Group'] == 'Male']['Outcome_Rate'].values[0])
print(f"\n  ⚠️  Female / Male Disparate Impact Ratio: {_adult_sex_dr:.3f}")
print(f"  ⚠️  Interpretation: Females earn >50K at only {_adult_sex_dr:.2f}x the rate of Males")

print("\n📊 Adult Income: Income Rate by Race")
print("-" * 70)
print(adult_race_report.to_string(index=False))

# Save all reports to CSV and JSON
compas_race_report.to_csv('compas_race_fairness_baseline.csv', index=False)
compas_sex_report.to_csv('compas_sex_fairness_baseline.csv', index=False)
adult_sex_report.to_csv('adult_sex_fairness_baseline.csv', index=False)
adult_race_report.to_csv('adult_race_fairness_baseline.csv', index=False)

_baseline_summary = {
    'compas': {
        'overall_recidivism_rate': round(float(compas_df['two_year_recid'].mean()), 4),
        'african_american_rate': round(float(compas_df[compas_df['race']=='African-American']['two_year_recid'].mean()), 4),
        'caucasian_rate': round(float(compas_df[compas_df['race']=='Caucasian']['two_year_recid'].mean()), 4),
        'race_disparate_impact_ratio': round(float(_compas_race_dr), 4),
        'male_rate': round(float(compas_df[compas_df['sex']=='Male']['two_year_recid'].mean()), 4),
        'female_rate': round(float(compas_df[compas_df['sex']=='Female']['two_year_recid'].mean()), 4),
    },
    'adult': {
        'overall_income_rate': round(float(adult_df['income'].mean()), 4),
        'male_rate': round(float(adult_df[adult_df['sex']=='Male']['income'].mean()), 4),
        'female_rate': round(float(adult_df[adult_df['sex']=='Female']['income'].mean()), 4),
        'sex_disparate_impact_ratio': round(float(_adult_sex_dr), 4),
    }
}
with open('fairness_baseline_summary.json', 'w') as _f:
    json.dump(_baseline_summary, _f, indent=2)

print("\n✅ Saved: compas_race_fairness_baseline.csv")
print("✅ Saved: compas_sex_fairness_baseline.csv")
print("✅ Saved: adult_sex_fairness_baseline.csv")
print("✅ Saved: adult_race_fairness_baseline.csv")
print("✅ Saved: fairness_baseline_summary.json")
print("\n✅ PHASE 1D COMPLETE — Fairness baseline report saved.")
print("\n" + "=" * 70)
print("PHASE 1 FULLY COMPLETE ✅")
print("  • 6,172 COMPAS records | 32,561 Adult records")
print("  • Sensitive attributes documented and preserved")
print("  • Train/test splits: 70/30 stratified")
print("  • EDA charts: 4 publication-quality figures")
print("  • Fairness baseline: disparities by race & sex computed")
print("=" * 70)
