# FairXplain: Explainable Fair AI in Indian Jurisprudence & Socioeconomics

**FairXplain** is a comprehensive machine learning fairness and explainability framework designed to audit and explain algorithmic bias in high-stakes public decision-making systems. The framework has been localized to evaluate pre-trial bail decisions under the **Indian Judicial System** (localized from the ProPublica COMPAS dataset) and socioeconomic classification under the **UCI Adult Income** dataset.

The system utilizes SHAP/LIME explanation models, multi-metric fairness audits (Demographic Parity, Equalized Odds), bias mitigation algorithms (Threshold Optimization), counterfactual recourse profile generation (DiCE), and causal graphical analysis (DoWhy).

---

## ⚖️ Indian Judicial Localization Mapping

The COMPAS pre-trial risk dataset and predictors have been fully mapped to Indian legal concepts and terminology:

| COMPAS / US Concept | Indian Judicial Concept | Description |
| :--- | :--- | :--- |
| **Recidivism Risk** | **Pre-Trial Detention / Repeat Offending Risk** | Prediction of probability of detention or repeat offense. |
| **Priors Count** | **Past FIRs** | Active criminal history recorded against the accused. |
| **Charge Degree (Felony/Misdemeanor)** | **IPC/BNS Offense Category (Cognizable vs. Non-Cognizable)** | Classification of offense seriousness (e.g. bailable vs. non-bailable). |
| **Days Screen to Arrest** | **Days from FIR Filing to Arrest** | Delay between FIR registration and actual police arrest. |
| **Length of Stay** | **Pre-trial Custody Duration (Days)** | Duration the accused has spent in pre-trial detention. |
| **Race (African-American / Caucasian)** | **Social Representation Category (SC/ST/OBC vs. General Category)** | Caste and community representation categories in demographic audits. |

---

## 🛠️ Main Features

1. **Space Navy Glassmorphic UI**: A premium, high-contrast dark theme dashboard styled with modern glassmorphism (`backdrop-filter: blur(16px)`), smooth animations, and glowing highlights.
2. **⚖️ NyayaAI Chatbot (Legal Audit)**: A conversational AI assistant that explains pre-trial detention predictions, SHAP coefficients, caste representation gaps, and DiCE recourse options based on active sliders.
3. **💼 VikasAI Chatbot (Socioeconomic Audit)**: A conversational AI assistant on the Adult Income page explaining income outcomes, gender pay gaps, and career recourse strategies.
4. **IPC/BNS Offense Preset Selector**: A dropdown to choose the type of case (Theft, Assault, Fraud, Cyber Crime, Murder) that updates a suggestions panel and auto-populates sliders with realistic legal parameters.
5. **PDF Report Generator**: Compiles all model metrics, scorecards, mitigation tradeoffs, and causal treatment effects into a publication-ready PDF.

---

## 📁 Repository Structure

- `FairXplain.ipynb` - Interactive Jupyter notebook of the pipeline.
- `fairxplain_pipeline.py` - Consolidated end-to-end python script running the complete 8-phase auditing pipeline.
- `app/` - Production deployment services:
  - `backend_app.py`: FastAPI backend on port `8000`. Serves master results, baseline models, fairness scorecards, predictions, NyayaAI/VikasAI chatbot APIs, and generates PDF reports.
  - `frontend_app.py`: HTML/JavaScript frontend on port `8080`. Renders the premium dashboard layout, interactive predictors, plots, and chats.
- `phases/` - Scripts representing each sequential phase of the auditing pipeline (Phase 1-8).

---

## 🚀 Running the Project

### 1. Prerequisites
Install all required packages:
```bash
pip install numpy pandas matplotlib xgboost shap lime tensorflow fairlearn dice-ml dowhy fpdf2 uvicorn fastapi watchfiles networkx
```

### 2. Run the Processing Pipeline
Execute the pipeline script to train the models and output auditing results (`.pkl` pickles, scorecard CSVs, causal results):
```bash
python fairxplain_pipeline.py
```

### 3. Start the Backend API (Port 8000)
Run the FastAPI uvicorn server:
```bash
python app/backend_app.py
```
*Healthcheck available at: `http://localhost:8000/api/health`*

### 4. Start the Frontend Dashboard (Port 8080)
Run the Uvicorn-hosted dashboard frontend:
```bash
python app/frontend_app.py
```
*Open your browser and navigate to: **[http://localhost:8080](http://localhost:8080)***
