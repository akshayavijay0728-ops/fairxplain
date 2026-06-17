import gradio as gr
import pandas as pd
import json
import os
import subprocess
import sys
import plotly.graph_objects as go
import plotly.express as px

# Configure stdout encoding on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths to generated files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR

def check_files_exist():
    required_files = [
        'compas_model_metrics.csv',
        'adult_model_metrics.csv',
        'mitigation_tradeoff.csv',
        'causal_fairness_results.json',
        'best_models.json',
        'compas_cleaned.csv',
        'adult_cleaned.csv',
        'fairness_scorecard.csv',
        'master_results.csv'
    ]
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in required_files)

def run_pipeline():
    pipeline_script = os.path.join(BASE_DIR, 'fairxplain_pipeline.py')
    print(f"Running pipeline script: {pipeline_script}")
    try:
        # Run pipeline script and capture output, setting environment to force UTF-8
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-u", pipeline_script],
            cwd=BASE_DIR,
            encoding="utf-8",
            capture_output=True,
            check=True,
            env=env
        )
        return "Success", f"Pipeline executed successfully!\n\n=== STDOUT ===\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return "Error", f"Pipeline execution failed!\n\n=== ERROR ===\n{e.stderr}\n\n=== STDOUT ===\n{e.stdout}"
    except Exception as e:
        return "Error", f"An unexpected error occurred: {str(e)}"

# Load data helper functions
def load_clean_data(dataset_name):
    filename = 'compas_cleaned.csv' if dataset_name == "COMPAS" else 'adult_cleaned.csv'
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def load_metrics():
    compas_path = os.path.join(DATA_DIR, 'compas_model_metrics.csv')
    adult_path = os.path.join(DATA_DIR, 'adult_model_metrics.csv')
    if not (os.path.exists(compas_path) and os.path.exists(adult_path)):
        return None, None
    compas_metrics = pd.read_csv(compas_path)
    adult_metrics = pd.read_csv(adult_path)
    return compas_metrics, adult_metrics

def load_mitigation():
    path = os.path.join(DATA_DIR, 'mitigation_tradeoff.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def load_causal():
    path = os.path.join(DATA_DIR, 'causal_fairness_results.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def load_scorecard():
    path = os.path.join(DATA_DIR, 'fairness_scorecard.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def load_master_results():
    path = os.path.join(DATA_DIR, 'master_results.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def load_counterfactuals(dataset_name):
    filename = 'compas_counterfactuals.json' if dataset_name == "COMPAS" else 'adult_counterfactuals.json'
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

# Visualization creators
def plot_eda_group_rates(dataset_name):
    df = load_clean_data(dataset_name)
    if df is None:
        return go.Figure().update_layout(title="No data. Run the pipeline first.")
    
    fig = go.Figure()
    if dataset_name == "COMPAS":
        # Recidivism rates by race
        rates = df.groupby('race')['two_year_recid'].mean().sort_values(ascending=True) * 100
        fig.add_trace(go.Bar(
            y=rates.index,
            x=rates.values,
            orientation='h',
            marker_color=['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#ffd400'],
            text=[f"{val:.1f}%" for val in rates.values],
            textposition='outside'
        ))
        fig.update_layout(
            title="COMPAS Recidivism Rate by Race",
            xaxis=dict(title='Recidivism Rate (%)', range=[0, 100]),
            yaxis=dict(title='Race')
        )
    else:
        # Income >50K rates by sex & top race
        rates_sex = df.groupby('sex')['income'].mean() * 100
        fig.add_trace(go.Bar(
            x=rates_sex.index,
            y=rates_sex.values,
            marker_color=['#FF9F9B', '#A1C9F4'],
            text=[f"{val:.1f}%" for val in rates_sex.values],
            textposition='outside'
        ))
        fig.update_layout(
            title="Adult Income >50K Rate by Sex",
            yaxis=dict(title='Percentage (%)', range=[0, 100]),
            xaxis=dict(title='Sex')
        )
        
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fbfbff')
    )
    return fig

def plot_model_comparison(compas_df, adult_df, dataset_name):
    df = compas_df if dataset_name == "COMPAS" else adult_df
    if df is None:
        return go.Figure().update_layout(title="No data. Run the pipeline first.")
    
    fig = go.Figure()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    colors = ['#A1C9F4', '#FFB482', '#8DE5A1']
    
    for idx, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=row['Model'],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        title=f"Baseline Model Performance - {dataset_name}",
        barmode='group',
        yaxis=dict(title='Value', range=[0, 1.05]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fbfbff')
    )
    return fig

def plot_mitigation_comparison():
    df = load_mitigation()
    if df is None:
        return go.Figure().update_layout(title="No data. Run the pipeline first.")
    
    fig = go.Figure()
    
    colors = ['#FF9F9B', '#A1C9F4', '#FFB482', '#8DE5A1']
    markers = ['circle', 'square', 'triangle-up', 'diamond']
    
    for idx, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[abs(row['DP_Diff'])],
            y=[row['Accuracy']],
            mode='markers+text',
            text=[row['Strategy']],
            textposition="top center",
            marker=dict(size=15, color=colors[idx % len(colors)], symbol=markers[idx % len(markers)]),
            name=row['Strategy']
        ))
    
    fig.update_layout(
        title="Accuracy vs. Demographic Parity Tradeoff (COMPAS)",
        xaxis=dict(title='|Demographic Parity Difference| (Lower is fairer)', range=[-0.05, 0.45]),
        yaxis=dict(title='Accuracy', range=[0.55, 0.85]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fbfbff')
    )
    return fig

def plot_causal_ate():
    data = load_causal()
    if data is None:
        return go.Figure().update_layout(title="No data. Run the pipeline first.")
    
    datasets = ['COMPAS Recidivism (Race: African-American)', 'Adult Income (Sex: Male)']
    ates = [data['compas']['ATE'], data['adult']['ATE']]
    
    fig = go.Figure(go.Bar(
        x=datasets,
        y=ates,
        marker_color=['#FF9F9B', '#A1C9F4'],
        text=[f"ATE: {ate:.4f}" for ate in ates],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Causal Average Treatment Effect (ATE) of Sensitive Attributes",
        yaxis=dict(title='Probability Shift (ATE)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fbfbff')
    )
    return fig

# UI loaders
def get_status_md():
    if check_files_exist():
        return "### 🟢 Status: Pipeline results found. Interactive dashboards are loaded!"
    else:
        return "### 🔴 Status: Pipeline results not found. Click 'Run Pipeline' below to generate everything."

def get_best_models_text():
    path = os.path.join(DATA_DIR, 'best_models.json')
    if not os.path.exists(path):
        return "🏆 **Best Models**: Run pipeline to evaluate."
    with open(path, 'r') as f:
        best = json.load(f)
    return f"🏆 **Best COMPAS Model**: `{best['compas']}` | 🏆 **Best Adult Model**: `{best['adult']}`"

def get_causal_text():
    data = load_causal()
    if data is None:
        return "Causal findings will appear here once the pipeline is run."
    return (
        f"💡 **COMPAS Inference**: {data['compas']['interpretation']}\n\n"
        f"💡 **Adult Inference**: {data['adult']['interpretation']}\n\n"
        f"📝 *Note: Causal estimates adjust for confounding factors (like age, priors, education) "
        f"using linear regression backdoor paths under the domain DAG assumptions.*"
    )

def get_counterfactuals_table(dataset_name):
    cfs = load_counterfactuals(dataset_name)
    if cfs is None:
        return pd.DataFrame({"Message": ["No counterfactuals. Run pipeline first."]})
    
    rows = []
    headers = ["Instance", "Feature Profile", "Values"]
    
    # Let's map features based on dataset
    if dataset_name == "COMPAS":
        feats = ["age", "charge_degree_enc", "priors_count", "days_b_screening_arrest", "length_of_stay"]
    else:
        feats = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
        
    for item in cfs[:2]: # Show first 2 instances
        inst_idx = item['instance']
        orig = item['original']
        # Show original row
        rows.append([f"Instance #{inst_idx} (Original)", ", ".join(feats), ", ".join(f"{v:.1f}" for v in orig)])
        # Show counterfactuals
        for c_idx, cf in enumerate(item['counterfactuals'][:2]):
            rows.append([f"Instance #{inst_idx} (CF {c_idx+1})", ", ".join(feats), ", ".join(f"{v:.1f}" for v in cf)])
            
    return pd.DataFrame(rows, columns=headers)

def refresh_dashboard():
    compas_metrics, adult_metrics = load_metrics()
    status_md = get_status_md()
    fig_eda_compas = plot_eda_group_rates("COMPAS")
    fig_eda_adult = plot_eda_group_rates("Adult")
    fig_model_compas = plot_model_comparison(compas_metrics, adult_metrics, "COMPAS")
    fig_model_adult = plot_model_comparison(compas_metrics, adult_metrics, "Adult Income")
    fig_mitigate = plot_mitigation_comparison()
    fig_causal = plot_causal_ate()
    
    best_models = get_best_models_text()
    causal_text = get_causal_text()
    
    scorecard = load_scorecard()
    if scorecard is None:
        scorecard = pd.DataFrame({"Message": ["No data"]})
        
    master = load_master_results()
    if master is None:
        master = pd.DataFrame({"Message": ["No data"]})
        
    compas_cf = get_counterfactuals_table("COMPAS")
    adult_cf = get_counterfactuals_table("Adult")
    
    return (
        status_md, best_models, causal_text,
        fig_eda_compas, fig_eda_adult,
        fig_model_compas, fig_model_adult,
        fig_mitigate, fig_causal,
        scorecard, master,
        compas_cf, adult_cf
    )

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown("# 📊 FairXplain: Explainable Fair AI Platform")
    gr.Markdown(
        "A comprehensive visualization platform showcasing machine learning model performance, "
        "algorithmic fairness audits, bias mitigation, local explainability, and causal inference "
        "on the ProPublica COMPAS Recidivism and UCI Adult Income datasets."
    )

    status_indicator = gr.Markdown(get_status_md())
    
    with gr.Row():
        btn_run = gr.Button("🚀 Run Pipeline (Train Models & Compute Metrics)", variant="primary")
        btn_refresh = gr.Button("🔄 Refresh Dashboard Data")

    pipeline_log = gr.Code(label="Pipeline Console output logs", interactive=False, lines=6)

    with gr.Tab("📋 Dashboard Overview"):
        gr.Markdown(
            "### Pipeline Execution Sequence & Components\n"
            "This pipeline executes the following 8 phases sequentially:\n"
            "1. **Phase 1A-1C**: Load, clean, and explore datasets (COMPAS & Adult Income).\n"
            "2. **Phase 1D**: Fairness Baseline Report (evaluating Disparate Impact).\n"
            "3. **Phase 2**: Baseline Model Training (Logistic Regression, Random Forest, XGBoost).\n"
            "4. **Phase 3**: Explainability (SHAP global importance and local LIME feature contributions).\n"
            "5. **Phase 4**: Fairness Audit (Demographic Parity, Equalized Odds metrics).\n"
            "6. **Phase 5**: Bias Mitigation (Reweighing, Exponentiated Gradient, Threshold Optimization).\n"
            "7. **Phase 6**: Counterfactual Explanations (DiCE counterfactual generation).\n"
            "8. **Phase 7**: Causal Fairness Analysis (DoWhy treatment effect estimation).\n"
            "9. **Phase 8**: Results Consolidation (Master results scorecard generation).\n"
        )
        best_models_md = gr.Markdown(get_best_models_text())

    with gr.Tab("📊 Phase 1: Exploratory Data Analysis"):
        gr.Markdown("#### Outgroup Disparities and Base Target Rates")
        with gr.Row():
            plot_eda1 = gr.Plot(label="COMPAS Outcome Rates")
            plot_eda2 = gr.Plot(label="Adult Income Outcome Rates")

    with gr.Tab("📈 Phase 2: Baseline Model Training"):
        with gr.Row():
            plot_model1 = gr.Plot(label="COMPAS Models")
            plot_model2 = gr.Plot(label="Adult Income Models")
            
    with gr.Tab("🔍 Phase 3 & 6: Local Explainability & Counterfactuals"):
        gr.Markdown(
            "### Actionable Counterfactual Explanations\n"
            "Displaying how a model's prediction would change under minimal modifications. "
            "Sensitive attributes (Race and Sex) are set as immutable to prevent generating unfair adjustments."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### COMPAS Recidivism Counterfactual Examples")
                cf_table_compas = gr.Dataframe(interactive=False)
            with gr.Column():
                gr.Markdown("#### Adult Income Counterfactual Examples")
                cf_table_adult = gr.Dataframe(interactive=False)

    with gr.Tab("⚖️ Phase 4 & 5: Fairness Audit & Mitigation"):
        gr.Markdown("### Fairness Audit Scorecard")
        scorecard_table = gr.Dataframe(interactive=False)
        
        gr.Markdown("### Tradeoff Analysis")
        plot_mitigate = gr.Plot(label="Accuracy vs. Demographic Parity Difference")

    with gr.Tab("🕸️ Phase 7: Causal Fairness"):
        gr.Markdown("### Causal Effects on Target Variables")
        with gr.Row():
            plot_causal = gr.Plot(label="Causal ATE")
            with gr.Column():
                gr.Markdown("#### Causal Interpretations")
                causal_interpretation_md = gr.Markdown(get_causal_text())

    with gr.Tab("🏆 Phase 8: Consolidated Master Results"):
        gr.Markdown("### Master Results Scorecard")
        master_results_table = gr.Dataframe(interactive=False)

    # Wire up run button
    def on_run_pipeline():
        status, log = run_pipeline()
        (
            status_md, best_models, causal_text,
            fig_eda_compas, fig_eda_adult,
            fig_model_compas, fig_model_adult,
            fig_mitigate, fig_causal,
            scorecard, master,
            compas_cf, adult_cf
        ) = refresh_dashboard()
        return (
            log, status_md, best_models, causal_text,
            fig_eda_compas, fig_eda_adult,
            fig_model_compas, fig_model_adult,
            fig_mitigate, fig_causal,
            scorecard, master,
            compas_cf, adult_cf
        )

    btn_run.click(
        fn=on_run_pipeline,
        inputs=[],
        outputs=[
            pipeline_log, status_indicator, best_models_md, causal_interpretation_md,
            plot_eda1, plot_eda2,
            plot_model1, plot_model2,
            plot_mitigate, plot_causal,
            scorecard_table, master_results_table,
            cf_table_compas, cf_table_adult
        ]
    )

    def on_refresh():
        return refresh_dashboard()

    btn_refresh.click(
        fn=on_refresh,
        inputs=[],
        outputs=[
            status_indicator, best_models_md, causal_interpretation_md,
            plot_eda1, plot_eda2,
            plot_model1, plot_model2,
            plot_mitigate, plot_causal,
            scorecard_table, master_results_table,
            cf_table_compas, cf_table_adult
        ]
    )

    # Initial load
    demo.load(
        fn=on_refresh,
        inputs=[],
        outputs=[
            status_indicator, best_models_md, causal_interpretation_md,
            plot_eda1, plot_eda2,
            plot_model1, plot_model2,
            plot_mitigate, plot_causal,
            scorecard_table, master_results_table,
            cf_table_compas, cf_table_adult
        ]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate"))