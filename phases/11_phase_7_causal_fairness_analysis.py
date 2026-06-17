"""
Name: Phase 7: Causal Fairness Analysis
Description: Applies causal inference using DoWhy to quantify treatment effects (race on recidivism, gender on income) via domain-specified DAGs and linear regression backdoor estimation. Visualizes causal graphs and outputs ATE estimates with fairness interpretations.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import dowhy
from dowhy import CausalModel
import json
import warnings
warnings.filterwarnings('ignore')

BG, FG, SG = '#1D1D20', '#fbfbff', '#909094'
C1, C2, C3, C4 = '#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'axes.edgecolor': SG, 'axes.labelcolor': FG,
                     'xtick.color': FG, 'ytick.color': FG,
                     'text.color': FG, 'grid.color': '#333337', 'grid.alpha': 0.4})

# ─────────────────────────────────────────────
# COMPAS CAUSAL DAG (Domain-specified)
# Causal assumptions:
#  race → priors_count (systemic policing bias)
#  race → two_year_recid (direct discrimination)
#  age  → priors_count (more time to accumulate)
#  age  → two_year_recid
#  priors_count → two_year_recid (mediator)
#  charge_degree_enc → two_year_recid
#  length_of_stay → two_year_recid
# ─────────────────────────────────────────────
print("=" * 65)
print("PHASE 7: CAUSAL FAIRNESS ANALYSIS")
print("=" * 65)

# Build COMPAS causal DataFrame
_compas_causal = compas_df[['age', 'priors_count', 'length_of_stay',
                              'two_year_recid']].copy()
_compas_causal['race_enc'] = (compas_df['race'] == 'African-American').astype(int)
_compas_causal['charge_enc'] = (compas_df['c_charge_degree'] == 'F').astype(int)

# DoWhy Causal Model — COMPAS
_compas_gml = """
graph [
  directed 1
  node [id "race_enc" label "race_enc"]
  node [id "age" label "age"]
  node [id "priors_count" label "priors_count"]
  node [id "charge_enc" label "charge_enc"]
  node [id "length_of_stay" label "length_of_stay"]
  node [id "two_year_recid" label "two_year_recid"]
  edge [source "race_enc" target "priors_count"]
  edge [source "race_enc" target "two_year_recid"]
  edge [source "age" target "priors_count"]
  edge [source "age" target "two_year_recid"]
  edge [source "priors_count" target "two_year_recid"]
  edge [source "charge_enc" target "two_year_recid"]
  edge [source "length_of_stay" target "two_year_recid"]
]
"""

_compas_model = CausalModel(
    data=_compas_causal,
    treatment='race_enc',
    outcome='two_year_recid',
    graph=_compas_gml
)

# Identify causal effect
_compas_identified = _compas_model.identify_effect(proceed_when_unidentifiable=True)
print("\nCOMPAS Estimand:")
print(str(_compas_identified)[:400])

# Estimate ATE via linear regression backdoor
_compas_estimate = _compas_model.estimate_effect(
    _compas_identified,
    method_name='backdoor.linear_regression',
    control_value=0, treatment_value=1,
    confidence_intervals=True, test_significance=True
)
_compas_ate = float(_compas_estimate.value)
print(f"\nCOMPAS ATE (race_enc → two_year_recid): {_compas_ate:.4f}")
print(f"  Interpretation: Being African-American increases recidivism probability")
print(f"  by {_compas_ate:.4f} after controlling for age, priors, charge degree")

# ─────────────────────────────────────────────
# ADULT CAUSAL DAG
# Causal assumptions:
#  sex → education_num (historical gender gap)
#  sex → income (direct gender pay gap)
#  age → education_num
#  age → income
#  education_num → income (mediator)
#  hours_per_week → income
#  capital_gain → income
# ─────────────────────────────────────────────
_adult_causal = adult_df[['age', 'education_num', 'hours_per_week',
                           'capital_gain', 'income']].copy()
_adult_causal['sex_enc'] = (adult_df['sex'] == 'Male').astype(int)

_adult_gml = """
graph [
  directed 1
  node [id "sex_enc" label "sex_enc"]
  node [id "age" label "age"]
  node [id "education_num" label "education_num"]
  node [id "hours_per_week" label "hours_per_week"]
  node [id "capital_gain" label "capital_gain"]
  node [id "income" label "income"]
  edge [source "sex_enc" target "education_num"]
  edge [source "sex_enc" target "income"]
  edge [source "age" target "education_num"]
  edge [source "age" target "income"]
  edge [source "education_num" target "income"]
  edge [source "hours_per_week" target "income"]
  edge [source "capital_gain" target "income"]
]
"""

_adult_model = CausalModel(
    data=_adult_causal,
    treatment='sex_enc',
    outcome='income',
    graph=_adult_gml
)
_adult_identified = _adult_model.identify_effect(proceed_when_unidentifiable=True)
_adult_estimate = _adult_model.estimate_effect(
    _adult_identified,
    method_name='backdoor.linear_regression',
    control_value=0, treatment_value=1,
    confidence_intervals=True, test_significance=True
)
_adult_ate = float(_adult_estimate.value)
print(f"\nAdult ATE (sex_enc=Male → income>50K): {_adult_ate:.4f}")
print(f"  Interpretation: Being Male increases >50K income probability")
print(f"  by {_adult_ate:.4f} after controlling for age, education, hours, capital_gain")

# ─────────────────────────────────────────────
# CAUSAL DAG VISUALIZATION
# ─────────────────────────────────────────────
causal_dag_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def draw_dag(ax, edges, node_colors, title):
    _G = nx.DiGraph()
    _G.add_edges_from(edges)
    _pos = nx.spring_layout(_G, seed=42, k=2)
    _colors = [node_colors.get(n, C1) for n in _G.nodes()]
    nx.draw_networkx_nodes(_G, _pos, ax=ax, node_color=_colors,
                           node_size=2000, alpha=0.9)
    nx.draw_networkx_labels(_G, _pos, ax=ax, font_size=8,
                            font_color=BG, font_weight='bold')
    nx.draw_networkx_edges(_G, _pos, ax=ax, edge_color=SG,
                           arrows=True, arrowsize=20,
                           connectionstyle='arc3,rad=0.1', width=1.5)
    ax.set_title(title, color=FG, fontsize=11, fontweight='bold')
    ax.set_facecolor(BG); ax.axis('off')

_compas_edges = [('race_enc', 'priors_count'), ('race_enc', 'two_year_recid'),
                 ('age', 'priors_count'), ('age', 'two_year_recid'),
                 ('priors_count', 'two_year_recid'), ('charge_enc', 'two_year_recid'),
                 ('length_of_stay', 'two_year_recid')]
_compas_nc = {'race_enc': C4, 'two_year_recid': C2,
              'priors_count': C3, 'age': C1,
              'charge_enc': C5, 'length_of_stay': C1}
draw_dag(ax1, _compas_edges, _compas_nc,
         'COMPAS Causal DAG\n(red=treatment, orange=outcome)')

_adult_edges = [('sex_enc', 'education_num'), ('sex_enc', 'income'),
                ('age', 'education_num'), ('age', 'income'),
                ('education_num', 'income'), ('hours_per_week', 'income'),
                ('capital_gain', 'income')]
_adult_nc = {'sex_enc': C4, 'income': C2, 'education_num': C3,
             'age': C1, 'hours_per_week': C1, 'capital_gain': C1}
draw_dag(ax2, _adult_edges, _adult_nc,
         'Adult Income Causal DAG\n(red=treatment, orange=outcome)')

causal_dag_fig.suptitle('FairXplain — Causal Fairness Graphs',
                         color=FG, fontsize=13, fontweight='bold')
causal_dag_fig.tight_layout()

# Save results
_causal_results = {
    'compas': {
        'treatment': 'race_enc (African-American=1)',
        'outcome': 'two_year_recid',
        'ATE': round(_compas_ate, 6),
        'interpretation': f'Being African-American increases recidivism prediction by {_compas_ate:.4f}',
        'mediators': ['priors_count'],
        'dag_assumption': 'race→priors_count→recidivism (indirect) + race→recidivism (direct)'
    },
    'adult': {
        'treatment': 'sex_enc (Male=1)',
        'outcome': 'income',
        'ATE': round(_adult_ate, 6),
        'interpretation': f'Being Male increases >50K probability by {_adult_ate:.4f}',
        'mediators': ['education_num'],
        'dag_assumption': 'sex→education→income (indirect) + sex→income (direct)'
    }
}
with open('causal_fairness_results.json', 'w') as _f:
    json.dump(_causal_results, _f, indent=2)

print("\n✅ Saved: causal_fairness_results.json")
print("✅ PHASE 7 COMPLETE — Causal fairness analysis done.")
print("\n⚠️  NOTE: Causal DAG structure reflects researcher-specified domain assumptions.")
print("   ATE estimates are conditional on these assumptions being valid.")
