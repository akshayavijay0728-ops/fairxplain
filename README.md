# FairXplain: Explainable Fair AI Framework

FairXplain: A comprehensive machine learning fairness and explainability framework that builds baseline models on COMPAS recidivism and Adult income datasets, then systematically audits them for bias using SHAP/LIME explanations, fairness metrics (demographic parity, equalized odds), and applies mitigation strategies (reweighing, exponentiated gradient, threshold optimization). The pipeline culminates in counterfactual explanations and causal fairness analysis to identify and quantify discriminatory patterns in high-stakes decision systems.

This project was exported from Zerve AI Gallery.

## Repository Structure

- `FairXplain.ipynb` - A Jupyter Notebook containing all steps as interactive cells. **(Recommended way to run)**
- `fairxplain_pipeline.py` - A single consolidated python script that runs the entire pipeline end-to-end.
- `phases/` - Directory containing each phase's code as a separate script (for inspection):
  - `00_check_package_dependencies.py`: Checks availability of 11 ML/AI packages (xgboost, shap, lime, aif360, tensorflow, fairlearn, dice_ml, dowhy, imblearn, fpdf2, streamlit) and reports which are installed vs. missing.
  - `01_phase_1a_load_and_clean_datasets.py`: Loads COMPAS recidivism and Adult income datasets from public sources, applies ProPublica-standard filters and cleaning (removing missing values, outliers, invalid entries), selects key features, and exports cleaned datasets as CSV files for fairness analysis.
  - `02_phase_1b_feature_preprocessing_and_splits.py`: Preprocesses COMPAS and Adult Income datasets for fairness analysis: applies feature engineering (age binning, encoding), removes sensitive attributes (race/sex) from features, scales/encodes features, performs stratified 70/30 train-test splits, and exports metadata.
  - `03_phase_1c_exploratory_data_analysis.py`: Creates 4 exploratory visualizations comparing recidivism & income outcomes by race/sex across COMPAS and Adult datasets, plus statistical summaries of disparities.
  - `04_phase_1d_fairness_baseline_report.py`: Analyzes fairness baseline disparities in COMPAS and Adult datasets by computing outcome rates, disparate impact ratios, and demographic breakdowns by race/sex. Generates detailed reports and summary metrics for bias assessment.
  - `05_phase_2_baseline_model_training.py`: Trains and evaluates three classification models (Logistic Regression, Random Forest, XGBoost) on COMPAS recidivism and Adult income datasets. Compares performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC), selects the best model by AUC for each dataset, and generates ROC curves and feature importance visualizations. Saves trained models and metrics tables.
  - `06_phase_3a_shap_explanations.py`: Computes SHAP explainability analysis for COMPAS and Adult Income models. Generates global feature importance bar charts and individual instance waterfall plots showing how features drive predictions (recidivism risk, income threshold) using Linear and Tree explainers.
  - `07_phase_3b_lime_explanations.py`: Generates LIME (Local Interpretable Model-agnostic Explanations) for individual predictions on COMPAS and Adult Income datasets, comparing top contributing features with SHAP explanations. Creates 3 visualizations per dataset showing positive/negative feature impacts for three case types (predicted positive, predicted negative, borderline).
  - `08_phase_4_fairness_audit.py`: Computes fairness metrics (demographic parity, equalized odds, equal opportunity) across COMPAS and Adult datasets for race/sex groups. Generates a bias scorecard table and heatmap visualization, saves results to CSV and JSON.
  - `09_phase_5_bias_mitigation.py`: Applies three bias mitigation strategies (Reweighing, Exponentiated Gradient, Threshold Optimizer) to a COMPAS model, comparing fairness metrics (demographic parity, equalized odds) against accuracy to visualize the fairness-accuracy tradeoff.
  - `10_phase_6_counterfactual_explanations.py`: Generates counterfactual explanations using DiCE for COMPAS recidivism and Adult income models. Shows what features would need to change to flip unfavorable predictions (e.g., from recidivist to non-recidivist) while keeping sensitive attributes immutable. Visualizes original vs. counterfactual feature values.
  - `11_phase_7_causal_fairness_analysis.py`: Applies causal inference using DoWhy to quantify treatment effects (race on recidivism, gender on income) via domain-specified DAGs and linear regression backdoor estimation. Visualizes causal graphs and outputs ATE estimates with fairness interpretations.
  - `12_phase_8_results_consolidation.py`: Consolidates FairXplain research results across all phases: compiles baseline model performance (ROC-AUC), fairness metrics (DP-Diff, EO-Diff) for COMPAS and Adult datasets, bias mitigation strategy tradeoffs, and causal ATEs into a master results table and publication-ready 4-panel summary visualization. Outputs narrative documenting key findings on fairness-accuracy tradeoffs and bias patterns.

- `app/` - Deployments:
  - `gradio_app.py`: Gradio dashboard application.
  - `fastapi_app.py`: FastAPI server backend.

## Prerequisites

Install the required packages:
```bash
pip install numpy pandas matplotlib xgboost shap lime tensorflow fairlearn dice-ml dowhy imbalanced-learn fpdf2 streamlit gradio uvicorn watchfiles networkx
```

## Running the Project

### Option 1: Jupyter Notebook (Recommended)
Open `FairXplain.ipynb` in VS Code or Jupyter Lab, and run the cells sequentially. This preserves in-memory variables between cells exactly like the Zerve canvas environment.

### Option 2: Command Line Pipeline
Run the consolidated pipeline script:
```bash
python fairxplain_pipeline.py
```

### Option 3: Deployments
Run the Gradio dashboard:
```bash
python app/gradio_app.py
```
Or run the FastAPI server:
```bash
uvicorn app.fastapi_app:app --reload
```
