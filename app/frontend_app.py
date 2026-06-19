from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FairXplain — Explainable &amp; Fair AI in Indian Jurisprudence</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet"/>
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root{
  --bg:#080C14; /* Deep Space Navy */
  --bg2:rgba(17, 24, 39, 0.7); /* Translucent Navy-Slate Glass */
  --bg3:#1F2937; /* Gray 800 Input BG */
  --border:rgba(255,255,255,0.08);
  --border-hover:#38BDF8; /* Sky Blue highlight */
  --fg:#FFFFFF; /* Pure White text */
  --sg:#94A3B8; /* Slate 400 - Muted Labels */
  --accent:#F59E0B; /* Vibrant Amber */
  --accent-gradient:linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
  --c1:#38BDF8; /* Sky Blue */
  --c2:#FB7185; /* Rose Red */
  --c3:#34D399; /* Emerald Green */
  --c4:#F87171; /* Coral Red */
  --c5:#C084FC; /* Light Purple */
}

*{box-sizing:border-box;margin:0;padding:0;}
body{
  background:var(--bg);
  background-image:radial-gradient(circle at 50% 0%, #1E1B4B 0%, var(--bg) 60%), radial-gradient(circle at 100% 100%, #030712 0%, var(--bg) 100%);
  color:var(--fg);
  font-family:'Outfit',system-ui,sans-serif;
  display:flex;
  min-height:100vh;
}

#sidebar{
  width:260px;
  min-height:100vh;
  background:#0E1322;
  padding:32px 0;
  position:fixed;
  left:0;
  top:0;
  z-index:100;
  border-right:1px solid var(--border);
}
#sidebar .logo{
  padding:0 24px 28px;
  border-bottom:1px solid var(--border);
}
#sidebar .logo h2{
  font-size:1.4rem;
  color:#FFFFFF;
  font-weight:800;
  letter-spacing:1px;
}
#sidebar .logo p{
  font-size:.78rem;
  color:var(--sg);
  margin-top:4px;
  font-weight:500;
}
#sidebar nav{
  padding:20px 12px;
}
#sidebar nav a{
  display:flex;
  align-items:center;
  gap:12px;
  padding:10px 14px;
  color:var(--sg);
  text-decoration:none;
  font-size:.88rem;
  font-weight:500;
  cursor:pointer;
  border-radius:8px;
  margin-bottom:4px;
  transition:all 0.2s ease;
}
#sidebar nav a:hover{
  background:rgba(255,255,255,0.04);
  color:#FFFFFF;
}
#sidebar nav a.active{
  background:rgba(56, 189, 248, 0.08);
  color:var(--c1);
  font-weight:600;
  box-shadow:inset 0 0 0 1px rgba(56, 189, 248, 0.2);
}
#sidebar nav a span.icon{
  font-size:1.1rem;
  width:24px;
  text-align:center;
}

#main{
  margin-left:260px;
  padding:40px 48px;
  flex:1;
  min-height:100vh;
}
.page{
  display:none;
  animation:fadeIn 0.4s ease forwards;
}
.page.active{
  display:block;
}
@keyframes fadeIn{
  from{opacity:0;transform:translateY(8px);}
  to{opacity:1;transform:translateY(0);}
}

.page-title{
  font-size:2.1rem;
  font-weight:800;
  color:#FFFFFF;
  margin-bottom:6px;
  letter-spacing:-0.5px;
}
.page-sub{
  color:var(--sg);
  font-size:.98rem;
  margin-bottom:32px;
}

.card{
  background:var(--bg2);
  backdrop-filter:blur(16px);
  -webkit-backdrop-filter:blur(16px);
  border-radius:18px;
  padding:28px;
  margin-bottom:24px;
  border:1px solid rgba(255, 255, 255, 0.07);
  box-shadow:0 8px 32px 0 rgba(0, 0, 0, 0.5);
  transition:all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.card:hover{
  border-color:rgba(56, 189, 248, 0.3);
  box-shadow:0 12px 40px 0 rgba(56, 189, 248, 0.1);
  transform:translateY(-2px);
}
.card h4{
  font-size:1.15rem;
  font-weight:700;
  margin-bottom:18px;
  color:#FFFFFF;
}

.stat-row{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:16px;
  margin-bottom:32px;
}
.stat{
  background:var(--bg2);
  border:1px solid var(--border);
  border-radius:14px;
  padding:24px 20px;
  text-align:center;
  position:relative;
  overflow:hidden;
  transition:transform 0.2s ease, border-color 0.25s ease;
}
.stat:hover{
  transform:translateY(-2px);
  border-color:var(--border-hover);
}
.stat::before{
  content:'';
  position:absolute;
  top:0;
  left:0;
  width:100%;
  height:4px;
  background:var(--c1);
}
.stat.c1::before{background:var(--c1);}
.stat.c2::before{background:var(--c2);}
.stat.c3::before{background:var(--c3);}
.stat.c4::before{background:var(--c4);}
.stat.c5::before{background:var(--c5);}
.stat.accent::before{background:var(--accent-gradient);}

.stat .val{
  font-size:2.1rem;
  font-weight:800;
  color:#FFFFFF;
  margin-bottom:4px;
}
.stat .lbl{
  font-size:.82rem;
  color:var(--sg);
  font-weight:600;
}

.badge-ok{
  background:rgba(52, 211, 153, 0.12);
  color:var(--c3);
  border:1px solid rgba(52, 211, 153, 0.2);
  border-radius:8px;
  padding:4px 12px;
  font-size:.8rem;
  font-weight:600;
}
.badge-warn{
  background:rgba(245, 158, 11, 0.12);
  color:var(--accent);
  border:1px solid rgba(245, 158, 11, 0.2);
  border-radius:8px;
  padding:4px 12px;
  font-size:.8rem;
  font-weight:600;
}
.badge-bad{
  background:rgba(248, 113, 113, 0.12);
  color:var(--c4);
  border:1px solid rgba(248, 113, 113, 0.2);
  border-radius:8px;
  padding:4px 12px;
  font-size:.8rem;
  font-weight:600;
}

table{
  width:100%;
  border-collapse:collapse;
  font-size:.9rem;
}
th{
  color:#FFFFFF;
  font-weight:700;
  text-align:left;
  padding:14px 16px;
  border-bottom:2px solid var(--border);
  background:rgba(255,255,255,0.02);
}
td{
  padding:14px 16px;
  border-bottom:1px solid rgba(255,255,255,0.04);
  color:var(--fg);
}
tr:hover td{
  background:rgba(255,255,255,0.03);
}

.chart-box{
  width:100%;
  min-height:360px;
}
.form-control,select{
  background:var(--bg3)!important;
  border:1px solid rgba(255,255,255,0.15)!important;
  color:#FFFFFF!important;
  border-radius:10px!important;
  padding:10px 14px!important;
}
.form-control:focus,select:focus{
  box-shadow:0 0 0 3px rgba(56, 189, 248, 0.2)!important;
  border-color:var(--c1)!important;
}
select option {
  background:var(--bg3);
  color:#FFFFFF;
}

.btn-primary{
  background:var(--accent-gradient);
  color:#000;
  border:none;
  font-weight:700;
  border-radius:10px;
  padding:12px 24px;
  transition:all 0.2s ease;
}
.btn-primary:hover{
  transform:translateY(-1px);
  box-shadow:0 4px 15px rgba(245, 158, 11, 0.4);
  color:#000;
}

.finding{
  background:rgba(255,255,255,0.05);
  color:#FFFFFF!important;
  border-left:4px solid var(--c1);
  padding:14px 18px;
  border-radius:0 12px 12px 0;
  margin-bottom:12px;
  font-size:.9rem;
  line-height:1.6;
}
.spinner{
  border:3px solid rgba(255,255,255,0.05);
  border-top:3px solid var(--c1);
  border-radius:50%;
  width:32px;
  height:32px;
  animation:spin .8s linear infinite;
  display:inline-block;
}
@keyframes spin{to{transform:rotate(360deg)}}

.tag{
  display:inline-block;
  background:rgba(255,255,255,0.04);
  border:1px solid var(--border);
  border-radius:8px;
  padding:4px 12px;
  font-size:.8rem;
  color:var(--c1);
  margin:3px;
  font-weight:600;
}

/* Beautiful Interactive Form elements */
.form-group{
  margin-bottom:20px;
}
.form-label-row{
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:6px;
}
.form-label{
  font-size:.88rem;
  color:#FFFFFF;
  font-weight:600;
}
.form-val-bubble{
  font-size:.85rem;
  color:var(--c1);
  font-weight:700;
  background:rgba(56, 189, 248, 0.08);
  padding:2px 8px;
  border-radius:6px;
}
input[type=range]{
  -webkit-appearance:none;
  width:100%;
  background:var(--bg3);
  height:6px;
  border-radius:3px;
  outline:none;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;
  appearance:none;
  width:16px;
  height:16px;
  border-radius:50%;
  background:var(--c1);
  cursor:pointer;
  box-shadow:0 0 8px rgba(56,189,248,0.5);
  transition:transform 0.15s ease;
}
input[type=range]::-webkit-slider-thumb:hover{
  transform:scale(1.25);
}

.preset-container{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-bottom:22px;
}
.preset-badge{
  background:rgba(255,255,255,0.03);
  border:1px solid var(--border);
  border-radius:8px;
  padding:6px 14px;
  font-size:.8rem;
  color:var(--sg);
  cursor:pointer;
  transition:all 0.2s ease;
  font-weight:600;
}
.preset-badge:hover{
  background:rgba(56,189,248,0.06);
  border-color:rgba(56,189,248,0.3);
  color:var(--c1);
  transform:translateY(-1px);
}

#pred-result, #a-pred-result{
  display:none;
  animation:fadeIn 0.3s ease forwards;
}
.risk-high{color:var(--c4);font-weight:800;font-size:1.45rem;}
.risk-low{color:var(--c3);font-weight:800;font-size:1.45rem;}

/* Chatbot Explicit Visibility CSS */
#compas-chat-box, #adult-chat-box {
  color: #FFFFFF !important;
  font-family: inherit;
}
#compas-chat-box *, #adult-chat-box * {
  color: #FFFFFF !important;
}
#compas-chat-box strong, #adult-chat-box strong {
  color: #FFFFFF !important;
  font-weight: 700 !important;
}
#compas-chat-box h5, #adult-chat-box h5 {
  font-weight: 700 !important;
  margin-top: 12px !important;
  margin-bottom: 6px !important;
  font-size: 0.95rem !important;
}
/* Chat Bubble Styles */
.chat-bubble-user {
  background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
  border: 1px solid rgba(56, 189, 248, 0.3) !important;
  border-radius: 14px 14px 0 14px !important;
  padding: 10px 14px !important;
  display: inline-block;
  max-width: 85%;
  color: #FFFFFF !important;
  text-align: left;
  box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2);
}
.chat-bubble-bot {
  background: rgba(255, 255, 255, 0.04) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 14px 14px 14px 0 !important;
  padding: 12px 16px !important;
  display: block;
  color: #FFFFFF !important;
  text-align: left;
  line-height: 1.6;
  margin-top: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.chat-bubble-bot-compas {
  border-left: 4px solid var(--c1) !important;
}
.chat-bubble-bot-adult {
  border-left: 4px solid var(--accent) !important;
}
</style>
</head>
<body>
<div id="sidebar">
  <div class="logo">
    <h2>⚖️ FAIRXPLAIN</h2>
    <p>Explainable &amp; Fair AI Framework</p>
  </div>
  <nav>
    <a class="active" onclick="nav('overview')"><span class="icon">🏠</span>Overview</a>
    
    <div style="font-size: .75rem; color: var(--sg); font-weight: 700; margin: 16px 16px 8px; letter-spacing: 0.5px; text-transform: uppercase;">Indian Judicial Audits</div>
    <a onclick="nav('compas-data')"><span class="icon">📊</span>Bail &amp; Custody Case Data</a>
    <a onclick="nav('compas-predict')"><span class="icon">🔮</span>Pre-Trial Detention Predictor</a>
    <a onclick="nav('compas-fairness')"><span class="icon">⚖️</span>Caste &amp; Gender Disparities</a>
    <a onclick="nav('compas-causal')"><span class="icon">🔗</span>Causal Justice &amp; Recourse</a>
    
    <div style="font-size: .75rem; color: var(--sg); font-weight: 700; margin: 16px 16px 8px; letter-spacing: 0.5px; text-transform: uppercase;">Adult Income</div>
    <a onclick="nav('adult-data')"><span class="icon">📊</span>Data &amp; Models</a>
    <a onclick="nav('adult-predict')"><span class="icon">🔮</span>Live Predictor</a>
    <a onclick="nav('adult-fairness')"><span class="icon">⚖️</span>Fairness Audit</a>
    <a onclick="nav('adult-causal')"><span class="icon">🔗</span>Causal &amp; Recourse</a>
    
    <div style="font-size: .75rem; color: var(--sg); font-weight: 700; margin: 16px 16px 8px; letter-spacing: 0.5px; text-transform: uppercase;">Executive</div>
    <a onclick="nav('report')"><span class="icon">📋</span>PDF Report</a>
  </nav>
</div>

<div id="main">

<!-- OVERVIEW -->
<div id="page-overview" class="page active">
  <div class="page-title">Indian Judicial &amp; Socioeconomic AI Audit</div>
  <div class="page-sub">Auditing algorithmic disparities in pre-trial detention risk models and income prediction classifiers</div>
  <div class="stat-row" id="ov-stats">
    <div class="stat c1"><div class="val">6,172</div><div class="lbl">Bail &amp; Custody Case Records</div></div>
    <div class="stat c2"><div class="val">32,561</div><div class="lbl">Adult Income Records</div></div>
    <div class="stat c3"><div class="val">0.731</div><div class="lbl">Best AUC (Judicial Model)</div></div>
    <div class="stat c5"><div class="val">0.928</div><div class="lbl">Best AUC (Adult)</div></div>
    <div class="stat c4"><div class="val">0.281</div><div class="lbl">Community Disparity (SC/ST/OBC)</div></div>
    <div class="stat accent"><div class="val">0.162</div><div class="lbl">Adult Sex DP-Diff</div></div>
  </div>
  <div class="card">
    <h4>🔑 Key Audit Findings (Indian Context Adaptations)</h4>
    <div id="ov-findings"></div>
  </div>
  <div class="card">
    <h4>📋 Research Question</h4>
    <p style="color:var(--fg);font-size:.95rem;line-height:1.7">
      Can machine learning classification systems be evaluated for systemic representation gaps and
      corrected using post-processing optimizations under standard legal auditing criteria (such as the 80% rule)?
    </p>
    <p style="margin-top:12px;color:var(--c3);font-weight:700;font-size:1.05rem;">Answer: Yes — with measurable performance tradeoffs.</p>
  </div>
  <div class="card">
    <h4>🗺️ System Architecture</h4>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:8px;">
      <div style="background:var(--bg3);border-radius:12px;padding:16px;border-top:2px solid var(--c1)">
        <div style="color:var(--c1);font-weight:600;margin-bottom:6px;">Phase 1–2</div>
        <div style="font-size:.82rem;color:var(--sg)">Data loading, cleaning, EDA, baseline model training (LR, RF, XGB)</div>
      </div>
      <div style="background:var(--bg3);border-radius:12px;padding:16px;border-top:2px solid var(--c2)">
        <div style="color:var(--c2);font-weight:600;margin-bottom:6px;">Phase 3</div>
        <div style="font-size:.82rem;color:var(--sg)">SHAP global &amp; local explanations, LIME instance-level analysis</div>
      </div>
      <div style="background:var(--bg3);border-radius:12px;padding:16px;border-top:2px solid var(--c3)">
        <div style="color:var(--c3);font-weight:600;margin-bottom:6px;">Phase 4–5</div>
        <div style="font-size:.82rem;color:var(--sg)">Fairness audit (5 metrics), bias mitigation (3 strategies)</div>
      </div>
      <div style="background:var(--bg3);border-radius:12px;padding:16px;border-top:2px solid var(--c5)">
        <div style="color:var(--c5);font-weight:600;margin-bottom:6px;">Phase 6–7</div>
        <div style="font-size:.82rem;color:var(--sg)">DiCE counterfactuals, DoWhy causal DAG &amp; ATE estimation</div>
      </div>
    </div>
  </div>
</div>

<!-- COMPAS DATA & MODELS -->
<div id="page-compas-data" class="page">
  <div class="page-title">📊 IPC/BNS Case Data &amp; Models</div>
  <div class="page-sub">Socio-demographic indicators, past criminal records, and baseline pre-trial detention risk prediction models in Indian courts</div>
  <div id="compas-ds-content"><div class="spinner"></div> Loading...</div>
</div>

<!-- COMPAS PREDICT -->
<div id="page-compas-predict" class="page">
  <div class="page-title">🔮 Pre-Trial Detention &amp; Bail Predictor</div>
  <div class="page-sub">Configure accused profile sliders to evaluate detention/bail risk and see live feature contributions</div>
  
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start;">
    <div class="card">
      <h4>Accused Case Profile Builder</h4>
      
      <div class="preset-container">
        <span style="font-size:.8rem;color:var(--sg);align-self:center;margin-right:4px;">Presets:</span>
        <div class="preset-badge" onclick="loadPreset(45, 0, 0, 0, 1)">🟢 Minor Bailable Offense (Low Detention Risk)</div>
        <div class="preset-badge" onclick="loadPreset(28, 2, 1, 0, 8)">🟡 Standard IPC/BNS Offense (Medium Detention Risk)</div>
        <div class="preset-badge" onclick="loadPreset(19, 9, 1, -2, 180)">🔴 Habitual Offender / Serious Offense (High Detention Risk)</div>
      </div>
      
      <div class="form-group" style="margin-top:16px;">
        <label class="form-label" style="margin-bottom:6px; display:block;">IPC/BNS Offense Category</label>
        <select id="p-case-type" class="form-control" onchange="onCaseTypeChange(this.value)">
          <option value="Theft">Theft / Burglary (IPC Sec 378 / BNS Sec 303)</option>
          <option value="Assault">Assault / Criminal Force (IPC Sec 323 / BNS Sec 115)</option>
          <option value="Fraud">Financial Fraud / Cheating (IPC Sec 420 / BNS Sec 318)</option>
          <option value="Cyber">Cyber Crime (IT Act Sec 66 / Sec 43)</option>
          <option value="Murder">Murder / Attempt to Murder (IPC Sec 302 / BNS Sec 101)</option>
        </select>
      </div>
      
      <div id="case-suggestions-box" style="background:rgba(56,189,248,0.03); border:1px dashed rgba(56,189,248,0.2); border-radius:10px; padding:12px 14px; margin-bottom:20px;">
        <div style="font-size:.82rem; font-weight:700; color:var(--c1); margin-bottom:4px;">💡 Suggested Accused Profile:</div>
        <div id="case-suggestions-text" style="font-size:.82rem; color:var(--sg); line-height:1.4; margin-bottom:10px;">
          Theft offenses are often cognizable but carry moderate sentences. Suggestion: Age: 28, Past FIRs: 1, Offense Type: Cognizable, Delay: 0, Custody Stay: 3 days.
        </div>
        <button class="btn btn-sm btn-outline-info" style="font-size:.78rem; font-weight:600; padding:4px 10px; border-radius:6px; background:transparent; border:1px solid var(--c1); color:var(--c1); cursor:pointer;" onclick="applyCaseSuggestion()">Apply Suggested Slider Profile</button>
      </div>
      
      <div class="form-group">
        <div class="form-label-row">
          <span class="form-label">Age of Accused</span>
          <span class="form-val-bubble" id="val-age">25</span>
        </div>
        <input id="p-age" type="range" min="18" max="80" value="25" oninput="upVal('age', this.value)"/>
      </div>
      
      <div class="form-group">
        <div class="form-label-row">
          <span class="form-label">Past FIRs Filed Against Accused</span>
          <span class="form-val-bubble" id="val-priors">2</span>
        </div>
        <input id="p-priors" type="range" min="0" max="30" value="2" oninput="upVal('priors', this.value)"/>
      </div>
      
      <div class="form-group">
        <div class="form-label-row">
          <span class="form-label">IPC/BNS Offense Type</span>
          <span class="form-val-bubble" id="val-charge">Cognizable Offense (Serious)</span>
        </div>
        <input id="p-charge" type="range" min="0" max="1" value="1" oninput="upVal('charge', this.value)"/>
      </div>
      
      <div class="form-group">
        <div class="form-label-row">
          <span class="form-label">FIR Registration to Arrest Delay (Days)</span>
          <span class="form-val-bubble" id="val-days">0</span>
        </div>
        <input id="p-days" type="range" min="-30" max="30" value="0" oninput="upVal('days', this.value)"/>
      </div>
      
      <div class="form-group">
        <div class="form-label-row">
          <span class="form-label">Duration of Pre-Trial Custody (Days)</span>
          <span class="form-val-bubble" id="val-los">5</span>
        </div>
        <input id="p-los" type="range" min="0" max="365" value="5" oninput="upVal('los', this.value)"/>
      </div>
      
      <button class="btn-primary w-100 mt-2" onclick="runPredict()">▶ Run Pre-Trial Detention Risk Evaluation</button>
    </div>
    
    <div>
      <div class="card" id="pred-result-card" style="min-height:240px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
        <div id="pred-idle-state">
          <div style="font-size:3rem; margin-bottom:14px; opacity:0.6;">🔮</div>
          <p style="color:var(--sg); font-size:.9rem; max-width:280px; margin:0 auto;">Modify features and click evaluation to view decision outcomes and explanation.</p>
        </div>
        
        <div id="pred-result" style="width:100%; text-align:left;">
          <h4 style="margin-bottom:12px;">Evaluation Outcome</h4>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div id="pred-label"></div>
            <div id="pred-prob-badge" style="font-weight:800; font-size:1.1rem; color:var(--accent);"></div>
          </div>
          <div id="pred-bar" style="background:var(--bg3); border-radius:6px; height:10px; overflow:hidden; margin-bottom:20px;">
            <div id="pred-bar-fill" style="height:100%; border-radius:6px; transition:.5s;"></div>
          </div>
          
          <h5 style="font-size:.9rem; font-weight:700; color:#FFFFFF; margin-bottom:12px;">Local Feature Contributions (Log-Odds Impact on Custody Risk)</h5>
          <div id="ch-pred-contrib" style="width:100%; min-height:240px; background:rgba(0,0,0,0.2); border-radius:12px; padding:10px;"></div>
          <p style="font-size:.8rem; color:var(--sg); line-height:1.5; margin-top:10px;">
            ℹ️ Green bars (negative values) decrease pre-trial detention risk probability, while red bars (positive values) increase it. This reflects coefficients of the active Logistic Regression model.
          </p>
        </div>
      </div>
      
      <!-- Legal NyayaAI Chatbot Card -->
      <div class="card" style="margin-top:24px; padding:24px;">
        <h4 style="display:flex; align-items:center; gap:8px;">⚖️ NyayaAI Chatbot Assistant</h4>
        <div id="compas-chat-box" style="height:250px; overflow-y:auto; background:rgba(0,0,0,0.25); border-radius:12px; padding:16px; margin-bottom:16px; border:1px solid var(--border);">
          <div style="margin-bottom:12px; font-size:.85rem; line-height:1.5;">
            <strong style="color:var(--c1);">NyayaAI:</strong> Welcome! I am NyayaAI, trained to explain our Indian Judicial Pre-Trial detention predictions and fairness scorecards. Ask me about the active case, caste representation gaps, or recourse options!
          </div>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px;">
          <span style="font-size:.78rem; color:var(--sg); align-self:center;">Suggested queries:</span>
          <button class="preset-badge" style="padding:4px 10px; font-size:.72rem; margin:0;" onclick="sendCompasChat('Why is the detention risk high/low for this profile?')">🔍 Why this prediction?</button>
          <button class="preset-badge" style="padding:4px 10px; font-size:.72rem; margin:0;" onclick="sendCompasChat('Explain the caste representation gaps and bias in this model.')">⚖️ Explain Caste Bias</button>
          <button class="preset-badge" style="padding:4px 10px; font-size:.72rem; margin:0;" onclick="sendCompasChat('What suggestions do you have to reduce/lower the predicted detention risk?')">🛠️ Recourse Suggestions</button>
        </div>
        <div style="display:flex; gap:10px;">
          <input id="compas-chat-input" type="text" class="form-control" style="flex:1; font-size:.88rem; padding:8px 12px!important;" placeholder="Type your legal audit question..." onkeydown="if(event.key==='Enter') sendCompasChat(this.value)"/>
          <button class="btn btn-primary" style="padding:8px 18px; font-size:.88rem;" onclick="sendCompasChat(document.getElementById('compas-chat-input').value)">Send</button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- COMPAS FAIRNESS -->
<div id="page-compas-fairness" class="page">
  <div class="page-title">⚖️ Community Representation Disparity Audit</div>
  <div class="page-sub">Auditing caste, community representation gaps, and gender biases in pre-trial detention risk models</div>
  <div id="compas-fairness-content"><div class="spinner"></div> Loading...</div>
</div>

<!-- COMPAS CAUSAL -->
<div id="page-compas-causal" class="page">
  <div class="page-title">🔗 Causal Jurisprudence &amp; Recourse</div>
  <div class="page-sub">DoWhy causal graphs, ATE values, and DiCE counterfactuals for the Pre-trial Bail &amp; Custody Model</div>
  <div id="compas-causal-content"><div class="spinner"></div> Loading...</div>
</div>

<!-- ADULT DATA & MODELS -->
<div id="page-adult-data" class="page">
  <div class="page-title">📊 Adult Income: Dataset &amp; Models</div>
  <div class="page-sub">Characteristics, sensitive attributes, and baseline models for UCI Adult Income dataset</div>
  <div id="adult-ds-content"><div class="spinner"></div> Loading...</div>
</div>

<!-- ADULT PREDICT -->
<div id="page-adult-predict" class="page">
  <div class="page-title">🔮 Adult Income Live Predictor</div>
  <div class="page-sub">Configure user socioeconomic sliders and selectors to evaluate probability of high income bracket (&gt;$50K/year)</div>
  
  <div style="display:grid;grid-template-columns:1.2fr 0.8fr;gap:24px;align-items:start;">
    <div class="card">
      <h4 style="margin-bottom:12px;">Adult Income Profile Builder</h4>
      
      <div class="preset-container">
        <span style="font-size:.8rem;color:var(--sg);align-self:center;margin-right:4px;">Presets:</span>
        <div class="preset-badge" onclick="loadAdultPreset(22, 9, 0, 0, 20, 'Private', 'HS-grad', 'Never-married', 'Other-service', 'Own-child')">🟢 Low Income</div>
        <div class="preset-badge" onclick="loadAdultPreset(38, 10, 0, 0, 40, 'Private', 'Some-college', 'Divorced', 'Sales', 'Unmarried')">🟡 Average Case</div>
        <div class="preset-badge" onclick="loadAdultPreset(45, 14, 15000, 0, 50, 'Private', 'Masters', 'Married-civ-spouse', 'Exec-managerial', 'Husband')">🔴 High Income</div>
      </div>
      
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
        <div>
          <div class="form-group">
            <div class="form-label-row"><span class="form-label">Age</span><span class="form-val-bubble" id="val-a-age">35</span></div>
            <input id="p-a-age" type="range" min="17" max="90" value="35" oninput="upAdultVal('age', this.value)"/>
          </div>
          <div class="form-group">
            <div class="form-label-row"><span class="form-label">Education (Years)</span><span class="form-val-bubble" id="val-a-education_num">10</span></div>
            <input id="p-a-education_num" type="range" min="1" max="16" value="10" oninput="upAdultVal('education_num', this.value)"/>
          </div>
          <div class="form-group">
            <div class="form-label-row"><span class="form-label">Capital Gain ($)</span><span class="form-val-bubble" id="val-a-capital_gain">0</span></div>
            <input id="p-a-capital_gain" type="range" min="0" max="99999" step="100" value="0" oninput="upAdultVal('capital_gain', this.value)"/>
          </div>
          <div class="form-group">
            <div class="form-label-row"><span class="form-label">Capital Loss ($)</span><span class="form-val-bubble" id="val-a-capital_loss">0</span></div>
            <input id="p-a-capital_loss" type="range" min="0" max="4356" step="50" value="0" oninput="upAdultVal('capital_loss', this.value)"/>
          </div>
          <div class="form-group">
            <div class="form-label-row"><span class="form-label">Hours Per Week</span><span class="form-val-bubble" id="val-a-hours_per_week">40</span></div>
            <input id="p-a-hours_per_week" type="range" min="1" max="99" value="40" oninput="upAdultVal('hours_per_week', this.value)"/>
          </div>
        </div>
        
        <div>
          <div class="form-group">
            <label class="form-label" style="margin-bottom:6px; display:block;">Workclass</label>
            <select id="p-a-workclass" class="form-control">
              <option value="Private">Private</option>
              <option value="Self-emp-not-inc">Self-emp-not-inc</option>
              <option value="Self-emp-inc">Self-emp-inc</option>
              <option value="Federal-gov">Federal-gov</option>
              <option value="Local-gov">Local-gov</option>
              <option value="State-gov">State-gov</option>
              <option value="Without-pay">Without-pay</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label" style="margin-bottom:6px; display:block;">Education Category</label>
            <select id="p-a-education" class="form-control">
              <option value="HS-grad">HS-grad</option>
              <option value="Some-college">Some-college</option>
              <option value="Bachelors">Bachelors</option>
              <option value="Masters">Masters</option>
              <option value="Doctorate">Doctorate</option>
              <option value="Assoc-voc">Assoc-voc</option>
              <option value="Assoc-acdm">Assoc-acdm</option>
              <option value="11th">11th</option>
              <option value="9th">9th</option>
              <option value="7th-8th">7th-8th</option>
              <option value="Prof-school">Prof-school</option>
              <option value="5th-6th">5th-6th</option>
              <option value="10th">10th</option>
              <option value="1st-4th">1st-4th</option>
              <option value="Preschool">Preschool</option>
              <option value="12th">12th</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label" style="margin-bottom:6px; display:block;">Marital Status</label>
            <select id="p-a-marital_status" class="form-control">
              <option value="Never-married">Never-married</option>
              <option value="Married-civ-spouse">Married-civ-spouse</option>
              <option value="Divorced">Divorced</option>
              <option value="Separated">Separated</option>
              <option value="Widowed">Widowed</option>
              <option value="Married-spouse-absent">Married-spouse-absent</option>
              <option value="Married-AF-spouse">Married-AF-spouse</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label" style="margin-bottom:6px; display:block;">Occupation</label>
            <select id="p-a-occupation" class="form-control">
              <option value="Prof-specialty">Prof-specialty</option>
              <option value="Craft-repair">Craft-repair</option>
              <option value="Exec-managerial">Exec-managerial</option>
              <option value="Adm-clerical">Adm-clerical</option>
              <option value="Sales">Sales</option>
              <option value="Other-service">Other-service</option>
              <option value="Machine-op-inspct">Machine-op-inspct</option>
              <option value="Transport-moving">Transport-moving</option>
              <option value="Handlers-cleaners">Handlers-cleaners</option>
              <option value="Farming-fishing">Farming-fishing</option>
              <option value="Tech-support">Tech-support</option>
              <option value="Protective-serv">Protective-serv</option>
              <option value="Priv-house-serv">Priv-house-serv</option>
              <option value="Armed-Forces">Armed-Forces</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label" style="margin-bottom:6px; display:block;">Relationship</label>
            <select id="p-a-relationship" class="form-control">
              <option value="Not-in-family">Not-in-family</option>
              <option value="Husband">Husband</option>
              <option value="Wife">Wife</option>
              <option value="Own-child">Own-child</option>
              <option value="Unmarried">Unmarried</option>
              <option value="Other-relative">Other-relative</option>
            </select>
          </div>
          <input id="p-a-native_country" type="hidden" value="United-States"/>
        </div>
      </div>
      
      <button class="btn-primary w-100 mt-2" onclick="runAdultPredict()">▶ Run Income Evaluation</button>
    </div>
    
    <div>
      <div class="card" id="a-pred-result-card" style="min-height:220px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
        <div id="a-pred-idle-state">
          <div style="font-size:3rem; margin-bottom:14px; opacity:0.6;">🔮</div>
          <p style="color:var(--sg); font-size:.9rem; max-width:280px; margin:0 auto;">Modify features and click evaluation to view predicted income bracket.</p>
        </div>
        
        <div id="a-pred-result" style="width:100%; text-align:left; display:none;">
          <h4 style="margin-bottom:12px;">Evaluation Outcome</h4>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div id="a-pred-label"></div>
            <div id="a-pred-prob-badge" style="font-weight:800; font-size:1.1rem; color:var(--accent);"></div>
          </div>
          <div id="a-pred-bar" style="background:var(--bg3); border-radius:6px; height:10px; overflow:hidden;">
            <div id="a-pred-bar-fill" style="height:100%; border-radius:6px; transition:.5s;"></div>
          </div>
          <p style="font-size:.8rem; color:var(--sg); line-height:1.5; margin-top:14px; margin-bottom:0;">
            This prediction represents the outcome of the baseline XGBoost model. A higher probability score indicates a higher likelihood of earning over $50,000 per year.
          </p>
        </div>
      </div>
      
      <!-- Adult VikasAI Chatbot Card -->
      <div class="card" style="margin-top:24px; padding:24px;">
        <h4 style="display:flex; align-items:center; gap:8px;">💼 VikasAI Chatbot Assistant</h4>
        <div id="adult-chat-box" style="height:250px; overflow-y:auto; background:rgba(0,0,0,0.25); border-radius:12px; padding:16px; margin-bottom:16px; border:1px solid var(--border);">
          <div style="margin-bottom:12px; font-size:.85rem; line-height:1.5;">
            <strong style="color:var(--accent);">VikasAI:</strong> Welcome! I am VikasAI, here to explain our socioeconomic income bracket classification predictions and gender equity scorecards. Ask me about the active profile, pay disparities, or career recourse!
          </div>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px;">
          <span style="font-size:.78rem; color:var(--sg); align-self:center;">Suggested queries:</span>
          <button class="preset-badge" style="padding:4px 10px; font-size:.72rem; margin:0;" onclick="sendAdultChat('Why is this profile predicted to earn <=50K or >50K?')">🔍 Why this outcome?</button>
          <button class="preset-badge" style="padding:4px 10px; font-size:.72rem; margin:0;" onclick="sendAdultChat('Explain the gender pay gap and bias metrics.')">⚖️ Explain Sex Bias</button>
          <button class="preset-badge" style="padding:4px 10px; font-size:.72rem; margin:0;" onclick="sendAdultChat('What suggestions do you have to improve this income profile?')">🛠️ Recourse Suggestions</button>
        </div>
        <div style="display:flex; gap:10px;">
          <input id="adult-chat-input" type="text" class="form-control" style="flex:1; font-size:.88rem; padding:8px 12px!important;" placeholder="Type your socioeconomic question..." onkeydown="if(event.key==='Enter') sendAdultChat(this.value)"/>
          <button class="btn btn-primary" style="padding:8px 18px; font-size:.88rem;" onclick="sendAdultChat(document.getElementById('adult-chat-input').value)">Send</button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ADULT FAIRNESS -->
<div id="page-adult-fairness" class="page">
  <div class="page-title">⚖️ Adult Income: Fairness Audit</div>
  <div class="page-sub">Auditing disparities, legal thresholds, and outgroup biases for UCI Adult Income dataset</div>
  <div id="adult-fairness-content"><div class="spinner"></div> Loading...</div>
</div>

<!-- ADULT CAUSAL -->
<div id="page-adult-causal" class="page">
  <div class="page-title">🔗 Adult Income: Causal &amp; Recourse</div>
  <div class="page-sub">Causal ATE (Sex → Income) adjusted for confounding paths and DiCE recourse scenarios</div>
  <div id="adult-causal-content"><div class="spinner"></div> Loading...</div>
</div>

<!-- PDF REPORT -->
<div id="page-report" class="page">
  <div class="page-title">📋 Executive Audit Report</div>
  <div class="page-sub">Generate and download a comprehensive PDF report summarizing all metrics and audits</div>
  <div class="card" style="max-width: 600px; padding: 40px 32px; text-align: center; margin: 40px auto;">
    <div style="font-size: 5rem; margin-bottom: 24px;">📄</div>
    <h4 style="font-size:1.35rem; margin-bottom:14px;">Download PDF Research Report</h4>
    <p style="color: var(--sg); font-size: 0.95rem; margin-bottom: 28px; line-height: 1.6; max-width:460px; margin-left:auto; margin-right:auto;">
      This publication-grade PDF compiles all exploratory data analysis, baseline performance tables,
      demographic parity / equalized odds scorecard heatmaps, bias mitigation tradeoffs,
      actionable counterfactual recourse profiles, and causal DAG treatment effects.
    </p>
    <a id="btn-download-pdf" target="_blank" class="btn btn-primary btn-lg" style="text-decoration: none; padding: 14px 40px; display: inline-block;">
      📥 Download PDF Audit Report
    </a>
  </div>
</div>

</div><!-- /main -->

<script>
const API = window.location.origin.replace(/:\d+$/,'') + ':8000';
const cache = {};

async function api(path){
  if(cache[path]) return cache[path];
  try{
    const r = await fetch(path.startsWith('http') ? path : API+path);
    const d = await r.json();
    cache[path] = d;
    return d;
  } catch(e){ return null; }
}

function nav(page){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('#sidebar nav a').forEach(a=>a.classList.remove('active'));
  document.getElementById('page-'+page).classList.add('active');
  const idx = ['overview','compas-data','compas-predict','compas-fairness','compas-causal','adult-data','adult-predict','adult-fairness','adult-causal','report'];
  
  // Custom sidebar active logic to skip dividers
  let links = document.querySelectorAll('#sidebar nav a');
  links.forEach(l => {
    let clickAttr = l.getAttribute('onclick');
    if (clickAttr && clickAttr.includes(`'${page}'`)) {
      l.classList.add('active');
    }
  });
  
  loaders[page]?.();
}

// Accused community category translations for Indian context
function translateRace(r){
  if(r === 'African-American') return 'Marginalized Communities (SC/ST/OBC)';
  if(r === 'Caucasian') return 'General Category';
  if(r === 'Hispanic') return 'Minority Communities (Oth)';
  if(r === 'Other') return 'Other Minority Communities';
  if(r === 'Asian') return 'Asian Minorities';
  if(r === 'Native American') return 'Tribal Communities (ST)';
  return r;
}

// Accused gender translations
function translateSex(s){
  if(s === 'Male') return 'Male';
  if(s === 'Female') return 'Female';
  return s;
}

// Interactive Predictor Helpers (COMPAS)
function upVal(type, val){
  const b = document.getElementById('val-'+type);
  if(type === 'charge'){
    b.textContent = val == 1 ? 'Cognizable Offense (Serious)' : 'Non-Cognizable Offense (Minor)';
  } else if(type === 'days'){
    b.textContent = val > 0 ? `+${val} days` : `${val} days`;
  } else {
    b.textContent = val;
  }
}

function loadPreset(age, priors, charge, days, los){
  document.getElementById('p-age').value = age;
  document.getElementById('p-priors').value = priors;
  document.getElementById('p-charge').value = charge;
  document.getElementById('p-days').value = days;
  document.getElementById('p-los').value = los;
  
  upVal('age', age);
  upVal('priors', priors);
  upVal('charge', charge);
  upVal('days', days);
  upVal('los', los);
  
  runPredict();
}

// Interactive Predictor Helpers (ADULT)
function upAdultVal(type, val){
  document.getElementById('val-a-'+type).textContent = val;
}

function loadAdultPreset(age, ed_num, cap_gain, cap_loss, hours, work, ed, marital, occ, rel){
  document.getElementById('p-a-age').value = age;
  document.getElementById('p-a-education_num').value = ed_num;
  document.getElementById('p-a-capital_gain').value = cap_gain;
  document.getElementById('p-a-capital_loss').value = cap_loss;
  document.getElementById('p-a-hours_per_week').value = hours;
  
  document.getElementById('p-a-workclass').value = work;
  document.getElementById('p-a-education').value = ed;
  document.getElementById('p-a-marital_status').value = marital;
  document.getElementById('p-a-occupation').value = occ;
  document.getElementById('p-a-relationship').value = rel;
  
  upAdultVal('age', age);
  upAdultVal('education_num', ed_num);
  upAdultVal('capital_gain', cap_gain);
  upAdultVal('capital_loss', cap_loss);
  upAdultVal('hours_per_week', hours);
  
  runAdultPredict();
}

// ── Overview ─────────────────────────────────────────────────────────────
async function loadOverview(){
  const d = await api(API+'/api/master-results');
  if(!d) return;
  const el = document.getElementById('ov-findings');
  
  // Custom Indian Judicial translations for key findings
  const findingsTranslations = {
    "COMPAS: LR best AUC=0.731 | XGB Adult best AUC=0.928": "Indian Judicial Model: Logistic Regression achieved best AUC (0.731) on pre-trial detention risk.",
    "SHAP: priors_count (0.528) is top COMPAS predictor — racially correlated proxy": "SHAP: Past FIRs (0.528) is the top predictor of detention risk — a community-correlated proxy variable.",
    "COMPAS Race: DP-Diff=0.281 — SEVERE racial disparity (exceeds 80% rule)": "Representation Gap: Caste/Community disparity is DP-Diff=0.281 — SEVERE caste disparity (exceeds 80% rule).",
    "Adult Sex: DP-Diff=0.162 — Women predicted high-income at 36% rate of men": "Adult Gender Pay Gap: DP-Diff=0.162 — Women favored at 36% the rate of men.",
    "Best mitigation: Threshold Optimizer (DP-Diff→0.278, Acc=0.648)": "Mitigation Tradeoff: Threshold Optimizer gives best tradeoff (DP-Diff reduced to 0.278, Accuracy maintained at 0.648).",
    "Causal ATE: Race→Recidivism=0.118, Sex→Income=0.173 (direct effects)": "Causal ATE: Caste Category &rarr; Custody has direct ATE=0.118, Gender &rarr; Income ATE=0.173 (direct effects)."
  };

  el.innerHTML = d.key_findings.map(f=>{
    const translatedText = findingsTranslations[f] || f;
    return `<div class="finding">• ${translatedText}</div>`;
  }).join('');
}

// ── COMPAS: Data & Models ────────────────────────────────────────────────
async function loadCompasData(){
  const el = document.getElementById('compas-ds-content');
  const d = await api(API+'/api/datasets');
  const m = await api(API+'/api/models');
  if(!d || !m){el.innerHTML='<p style="color:#f04438">Could not load — ensure Backend API is running.</p>';return;}

  const rawRaceLabels = Object.keys(d.compas.race_distribution);
  const raceLabels = rawRaceLabels.map(translateRace);
  const raceVals   = Object.values(d.compas.race_distribution);
  
  const rawSexLabels  = Object.keys(d.compas.sex_distribution);
  const sexLabels  = rawSexLabels.map(translateSex);
  const sexVals    = Object.values(d.compas.sex_distribution);

  function tableHtml(rows, best){
    return `<table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th></th></tr></thead><tbody>`+
    rows.map(r=>`<tr>
      <td><strong>${r.Model}</strong></td>
      <td>${(+r.Accuracy).toFixed(3)}</td>
      <td>${(+r.Precision).toFixed(3)}</td>
      <td>${(+r.Recall).toFixed(3)}</td>
      <td>${(+r.F1).toFixed(3)}</td>
      <td>${(+r['ROC-AUC']).toFixed(3)}</td>
      <td>${r.Model===best?'<span class="badge-ok">✓ Best</span>':''}</td>
    </tr>`).join('')+`</tbody></table>`;
  }

  el.innerHTML = `
  <div class="stat-row">
    <div class="stat c1"><div class="val">${d.compas.records.toLocaleString()}</div><div class="lbl">Total Case Records</div></div>
    <div class="stat c3"><div class="val">${(d.compas.positive_rate*100).toFixed(1)}%</div><div class="lbl">Pre-trial Detention Rate</div></div>
    <div class="stat c2"><div class="val">${d.compas.features}</div><div class="lbl">Judicial Features</div></div>
    <div class="stat c5"><div class="val">0.731</div><div class="lbl">Best Model ROC-AUC</div></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
    <div class="card"><h4>Accused — Caste/Community Distribution</h4><div id="ch-compas-race" class="chart-box"></div></div>
    <div class="card"><h4>Accused — Gender Distribution</h4><div id="ch-compas-sex" class="chart-box"></div></div>
  </div>
  <div class="card" style="margin-top:24px;">
    <h4>Baseline Models Evaluation (Pre-trial Detention Risk)</h4>
    ${tableHtml(m.compas.results, m.compas.best_model)}
    <div id="ch-compas-models" class="chart-box" style="min-height:280px;margin-top:24px;"></div>
  </div>
  <div class="card">
    <h4>Sensitive Attributes Treatment</h4>
    <p style="font-size:.9rem;color:var(--sg);line-height:1.7;margin:0;">
      <strong style="color:var(--fg)">Indian Judicial Auditing:</strong> protected attributes like Caste/Community (SC/ST/OBC) and Gender are excluded from the pre-trial bail models' features. However, proxy variables like the length of custody and past FIR registration delays retain indirect disparities.
    </p>
  </div>`;

  const cfg = {
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Outfit', color:'#F8FAFC',size:11},
    margin:{t:30,b:40,l:50,r:20},
    showlegend:false,
    xaxis:{gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}},
    yaxis:{gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}}
  };
  const colors = ['#38BDF8','#FB7185','#34D399','#F87171','#C084FC','#F59E0B'];

  Plotly.newPlot('ch-compas-race',[{type:'bar',x:raceLabels,y:raceVals,
    marker:{color:colors}}],cfg);
  Plotly.newPlot('ch-compas-sex',[{type:'pie',labels:sexLabels,values:sexVals,
    marker:{colors:['#38BDF8','#F87171']},hole:.4,textinfo:'label+percent'}],
    {paper_bgcolor:'rgba(0,0,0,0)',font:{family:'Outfit', color:'#F8FAFC'},margin:{t:10,b:10,l:10,r:10}});

  // Models bar chart
  const mCfg = {
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Outfit', color:'#F8FAFC'},
    margin:{t:30,b:30,l:60,r:20},
    barmode:'group',
    yaxis:{range:[0,1],gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}},
    xaxis:{gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}},
    legend:{orientation:'h', y:-0.2, font:{color:'#94A3B8'}}
  };
  const metrics=['Accuracy','F1','ROC-AUC'];
  const traces = metrics.map((mt,i)=>({name:mt,type:'bar',x:m.compas.results.map(r=>r.Model),
    y:m.compas.results.map(r=>+(r[mt]||r['ROC-AUC']||0).toFixed(3)),marker:{color:['#38BDF8','#FB7185','#34D399'][i]}}));
  Plotly.newPlot('ch-compas-models',traces,mCfg);
}

// ── COMPAS: Predict ──────────────────────────────────────────────────────
async function loadCompasPredict(){}

// ── COMPAS: Fairness ─────────────────────────────────────────────────────
async function loadCompasFairness(){
  const el = document.getElementById('compas-fairness-content');
  const d = await api(API+'/api/fairness');
  const m = await api(API+'/api/mitigation');
  if(!d || !m){el.innerHTML='<p style="color:#f04438">Backend not reachable.</p>';return;}

  function badge(v){
    const n=parseFloat(v);
    if(n>0.2) return `<span class="badge-bad">${n.toFixed(4)} ⚠️</span>`;
    if(n>0.1) return `<span class="badge-warn">${n.toFixed(4)} ⚡</span>`;
    return `<span class="badge-ok">${n.toFixed(4)} ✓</span>`;
  }

  // Filter scorecard for COMPAS and map headers
  const sc = d.scorecard;
  const filteredSc = sc.filter(r => r.Metric && !r.Metric.includes('Adult') && (r['COMPAS (Race)'] !== undefined));

  // Translate metric rows
  const rowNameTranslations = {
    "Demographic Parity Difference": "Demographic Parity Gap (SC/ST/OBC vs General)",
    "Demographic Parity Ratio": "Demographic Parity Ratio",
    "Equalized Odds Difference": "Equalized Odds Gap (Accuracy Disparity)",
    "Equal Opportunity Difference": "Equal Opportunity Gap (Bail False Rejections)"
  };

  el.innerHTML=`
  <div class="card">
    <h4>Caste &amp; Gender Representation Scorecard</h4>
    <div style="overflow-x:auto;">
      <table><thead><tr><th>Evaluation Metric</th><th>Community Gap (SC/ST/OBC)</th><th>Gender Gap (Female/Male)</th></tr></thead>
      <tbody>${filteredSc.map(r=>`<tr><td><strong>${rowNameTranslations[r.Metric] || r.Metric}</strong></td>
        <td>${badge(r['COMPAS (Race)'])}</td><td>${badge(r['COMPAS (Sex)'])}</td></tr>`).join('')}
      </tbody></table>
    </div>
    <p style="margin-top:16px;font-size:.8rem;color:var(--sg)">
      🔴 &gt;0.2 = Violates 80% rule (actionable bias) &nbsp;|&nbsp;
      🟡 0.1–0.2 = Disparity &nbsp;|&nbsp; 🟢 &lt;0.1 = Parity
    </p>
  </div>
  <div class="card">
    <h4>Pre-trial Custody Bias Mitigation (Indian Judicial context)</h4>
    <div style="overflow-x:auto;">
      <table><thead><tr><th>Strategy</th><th>Accuracy</th><th>F1</th><th>Community DP-Diff</th><th>Community EO-Diff</th><th>Verdict</th></tr></thead>
      <tbody>${m.strategies.map(r=>{
        const isBase=r.Strategy==='Baseline';
        const isBest=r.Strategy==='Threshold Optimizer';
        return `<tr><td><strong>${r.Strategy}</strong></td>
          <td>${(+r.Accuracy).toFixed(3)}</td><td>${(+r.F1).toFixed(3)}</td>
          <td>${(+r.DP_Diff).toFixed(4)}</td><td>${(+r.EO_Diff).toFixed(4)}</td>
          <td>${isBase?'<span class="badge-bad">Unfair</span>':isBest?'<span class="badge-ok">✓ Best</span>':'<span class="badge-warn">Tradeoff</span>'}</td>
        </tr>`;}).join('')}</tbody></table>
    </div>
  </div>
  <div class="card"><h4>Caste/Community Parity vs Accuracy Tradeoff</h4><div id="ch-compas-tradeoff" class="chart-box"></div></div>`;

  Plotly.newPlot('ch-compas-tradeoff',[{
    type:'scatter',mode:'markers+text',
    x:m.strategies.map(r=>+r.DP_Diff),y:m.strategies.map(r=>+r.Accuracy),
    text:m.strategies.map(r=>r.Strategy),textposition:'top center',
    marker:{size:16,color:['#F87171','#94A3B8','#38BDF8','#34D399']},
    textfont:{color:'#FFFFFF', family:'Outfit'}
  }],{
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Outfit', color:'#F8FAFC'},
    margin:{t:20,b:50,l:60,r:20},
    xaxis:{title:'Community DP Difference (lower=fairer)',gridcolor:'rgba(255,255,255,0.06)',linecolor:'rgba(255,255,255,0.1)', range:[-0.02, 0.45], tickfont:{color:'#94A3B8'}},
    yaxis:{title:'Accuracy',gridcolor:'rgba(255,255,255,0.06)',linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}},
    shapes:[{type:'line',x0:0.2,x1:0.2,y0:0,y1:1,line:{color:'#F59E0B',dash:'dot',width:2}}],
    annotations:[{x:0.2,y:0.55,text:'80% Rule Threshold',font:{color:'#F59E0B', family:'Outfit'},showarrow:false}]
  });
}

// ── COMPAS: Causal ───────────────────────────────────────────────────────
async function loadCompasCausal(){
  const el = document.getElementById('compas-causal-content');
  const cfs = await api(API+'/api/counterfactuals');
  const d = await api(API+'/api/causal');
  if(!cfs || !d){el.innerHTML='<p style="color:#f04438">Backend not reachable.</p>';return;}

  const namesMap = {
    'age': 'Age of Accused',
    'charge_degree_enc': 'Offense Type (Cognizable)',
    'priors_count': 'Past FIRs',
    'days_b_screening_arrest': 'Days from FIR to Arrest',
    'length_of_stay': 'Pre-trial Custody Duration (Days)'
  };

  function cfTable(rows){
    return rows.map((cf,i)=>`
      <div style="margin-bottom:18px;padding:18px;background:rgba(255,255,255,0.01);border-radius:12px;border:1px solid var(--border);">
        <div style="font-size:.82rem;color:var(--sg);margin-bottom:12px;font-weight:600;">Accused Profile #${i+1}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;font-size:.85rem;">
          <div style="background:rgba(248,113,113,0.03); padding:14px; border-radius:8px; border:1px solid rgba(248,113,113,0.08);">
            <div style="color:var(--c4);font-weight:700;margin-bottom:8px;font-size:.9rem;">Original Profile (Detention Risk: High)</div>
            ${Object.entries(cf.original||{}).map(([k,v])=>`<div>${namesMap[k] || k}: <span style="color:var(--fg);font-weight:600;">${v}</span></div>`).join('')}
          </div>
          <div style="background:rgba(52,211,153,0.03); padding:14px; border-radius:8px; border:1px solid rgba(52,211,153,0.08);">
            <div style="color:var(--c3);font-weight:700;margin-bottom:8px;font-size:.9rem;">Actionable Recourse (Detention Risk: Low)</div>
            ${Object.entries(cf.counterfactual||{}).map(([k,v])=>`<div>${namesMap[k] || k}: <span style="color:var(--c1);font-weight:600;">${v}</span></div>`).join('')}
          </div>
        </div>
      </div>`).join('');
  }

  el.innerHTML=`
  <div class="card">
    <h4>Causal Effect (Social Category &rarr; Custody Outcome)</h4>
    <div class="stat c4" style="margin-bottom:18px; max-width: 320px;"><div class="val">${d.compas.ate}</div><div class="lbl">Average Treatment Effect (ATE)</div></div>
    <p style="font-size:.88rem;color:var(--sg);line-height:1.7;margin:0;">Socio-demographic Category (SC/ST/OBC) has a direct ATE = ${d.compas.ate} on detention predictions even after controlling for age, prior record, and offense classification.</p>
    <div id="ch-compas-causal-ate" class="chart-box" style="min-height:220px;margin-top:18px;"></div>
  </div>
  <div class="card">
    <h4>Causal Structural DAG (Indian Court Custody Assumptions)</h4>
    <div id="ch-dag" class="chart-box" style="min-height:320px;"></div>
  </div>
  <div class="card">
    <h4>Actionable Bail Recourse (DiCE Counterfactuals)</h4>
    ${cfTable(cfs.compas)}
  </div>`;

  const cfg={
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Outfit', color:'#F8FAFC'},
    margin:{t:30,b:30,l:40,r:20},
    xaxis:{gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}},
    yaxis:{gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}}
  };

  Plotly.newPlot('ch-compas-causal-ate',[
    {type:'bar',x:['Without Social Category Control','With Community Category Control (ATE)'],y:[0.281,0.118],
      marker:{color:['#F87171','#38BDF8']},text:['0.281','0.118'],textposition:'outside'}],
    {...cfg,yaxis:{range:[0,.4]},title:{text:'Community Disparity Before/After Causal Control',font:{family:'Outfit',size:13,color:'#fff'}}});

  // DAG Diagram
  const nodes=['Community','Age','Past FIRs','IPC/BNS Class','Detention Risk'];
  const x=[0,1,2,1,3]; const y=[2,3,2,1,2];
  const edges=[[0,2],[0,4],[1,2],[1,4],[2,4],[3,4]];
  const traces=[{type:'scatter',mode:'markers+text',x,y,text:nodes,textposition:'top center',
    marker:{size:28,color:['#F87171','#38BDF8','#FB7185','#34D399','#C084FC'],
    line:{color:'#fff',width:1.5}},textfont:{family:'Outfit',size:11,color:'#fff'}}];
  const shapes=edges.map(([a,b])=>({type:'line',x0:x[a],y0:y[a],x1:x[b],y1:y[b],
    line:{color:'#94A3B8',width:1.5}}));
  Plotly.newPlot('ch-dag',traces,{...cfg,xaxis:{visible:false},yaxis:{visible:false},shapes,
    margin:{t:10,b:10,l:10,r:10}});
}

// ── ADULT: Data & Models ─────────────────────────────────────────────────
async function loadAdultData(){
  const el = document.getElementById('adult-ds-content');
  const d = await api(API+'/api/datasets');
  const m = await api(API+'/api/models');
  if(!d || !m){el.innerHTML='<p style="color:#f04438">Could not load dataset metrics.</p>';return;}

  const sexLabels = Object.keys(d.adult.sex_distribution);
  const sexVals   = Object.values(d.adult.sex_distribution);
  const raceLabels = Object.keys(d.adult.race_distribution);
  const raceVals   = Object.values(d.adult.race_distribution);

  function tableHtml(rows, best){
    return `<table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th></th></tr></thead><tbody>`+
    rows.map(r=>`<tr>
      <td><strong>${r.Model}</strong></td>
      <td>${(+r.Accuracy).toFixed(3)}</td>
      <td>${(+r.Precision).toFixed(3)}</td>
      <td>${(+r.Recall).toFixed(3)}</td>
      <td>${(+r.F1).toFixed(3)}</td>
      <td>${(+r['ROC-AUC']).toFixed(3)}</td>
      <td>${r.Model===best?'<span class="badge-ok">✓ Best</span>':''}</td>
    </tr>`).join('')+`</tbody></table>`;
  }

  el.innerHTML = `
  <div class="stat-row">
    <div class="stat c2"><div class="val">${d.adult.records.toLocaleString()}</div><div class="lbl">Adult Records</div></div>
    <div class="stat accent"><div class="val">${(d.adult.positive_rate*100).toFixed(1)}%</div><div class="lbl">High Income Rate (&gt;50K)</div></div>
    <div class="stat c3"><div class="val">14</div><div class="lbl">Features Count</div></div>
    <div class="stat c5"><div class="val">0.928</div><div class="lbl">Best ROC-AUC</div></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
    <div class="card"><h4>Adult — Sex Distribution</h4><div id="ch-adult-sex" class="chart-box"></div></div>
    <div class="card"><h4>Adult — Race Distribution</h4><div id="ch-adult-race" class="chart-box"></div></div>
  </div>
  <div class="card" style="margin-top:24px;">
    <h4>Baseline Models Evaluation</h4>
    ${tableHtml(m.adult.results, m.adult.best_model)}
    <div id="ch-adult-models" class="chart-box" style="min-height:280px;margin-top:24px;"></div>
  </div>`;

  const cfg = {
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Outfit', color:'#F8FAFC',size:11},
    margin:{t:30,b:40,l:50,r:20},
    showlegend:false,
    xaxis:{gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}},
    yaxis:{gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}}
  };
  const colors = ['#38BDF8','#FB7185','#34D399','#F87171','#C084FC'];

  Plotly.newPlot('ch-adult-race',[{type:'bar',x:raceLabels,y:raceVals,
    marker:{color:colors}}],cfg);
  Plotly.newPlot('ch-adult-sex',[{type:'pie',labels:sexLabels,values:sexVals,
    marker:{colors:['#38BDF8','#F87171']},hole:.4,textinfo:'label+percent'}],
    {paper_bgcolor:'rgba(0,0,0,0)',font:{family:'Outfit', color:'#F8FAFC'},margin:{t:10,b:10,l:10,r:10}});

  // Models bar chart
  const mCfg = {
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Outfit', color:'#F8FAFC'},
    margin:{t:30,b:30,l:60,r:20},
    barmode:'group',
    yaxis:{range:[0,1],gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}},
    xaxis:{gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}},
    legend:{orientation:'h', y:-0.2, font:{color:'#94A3B8'}}
  };
  const metrics=['Accuracy','F1','ROC-AUC'];
  const traces = metrics.map((mt,i)=>({name:mt,type:'bar',x:m.adult.results.map(r=>r.Model),
    y:m.adult.results.map(r=>+(r[mt]||r['ROC-AUC']||0).toFixed(3)),marker:{color:['#38BDF8','#FB7185','#34D399'][i]}}));
  Plotly.newPlot('ch-adult-models',traces,mCfg);
}

// ── ADULT: Predict ───────────────────────────────────────────────────────
async function loadAdultPredict(){}

// ── ADULT: Fairness ──────────────────────────────────────────────────────
async function loadAdultFairness(){
  const el = document.getElementById('adult-fairness-content');
  const d = await api(API+'/api/fairness');
  if(!d){el.innerHTML='<p style="color:#f04438">Backend not reachable.</p>';return;}

  function badge(v){
    const n=parseFloat(v);
    if(n>0.2) return `<span class="badge-bad">${n.toFixed(4)} ⚠️</span>`;
    if(n>0.1) return `<span class="badge-warn">${n.toFixed(4)} ⚡</span>`;
    return `<span class="badge-ok">${n.toFixed(4)} ✓</span>`;
  }

  // Filter scorecard for Adult
  const sc = d.scorecard;
  const filteredSc = sc.filter(r => r.Metric && (r.Metric.includes('Adult') || r['Adult (Sex)'] !== undefined));

  el.innerHTML=`
  <div class="card">
    <h4>Adult Income Fairness Scorecard</h4>
    <div style="overflow-x:auto;">
      <table><thead><tr><th>Metric</th><th>Adult (Sex)</th><th>Adult (Race)</th></tr></thead>
      <tbody>${filteredSc.map(r=>`<tr><td><strong>${r.Metric}</strong></td>
        <td>${badge(r['Adult (Sex)'])}</td><td>${badge(r['Adult (Race)'])}</td></tr>`).join('')}
      </tbody></table>
    </div>
    <p style="margin-top:16px;font-size:.8rem;color:var(--sg)">
      🔴 &gt;0.2 = Violates 80% rule (actionable bias) &nbsp;|&nbsp;
      🟡 0.1–0.2 = Concerning disparity &nbsp;|&nbsp; 🟢 &lt;0.1 = Acceptable parity
    </p>
  </div>
  <div class="card">
    <h4>Gender Pay &amp; Income Disparities</h4>
    <div class="finding">🟡 <strong>Adult Sex DP-Diff=0.162</strong> — Women predicted high-income at only 36% the rate of men. HIGH disparity.</div>
    <div class="finding">✅ <strong>Adult Sex EO-Diff=0.057</strong> — Equalized odds nearly satisfied for Adult income sex gap.</div>
  </div>`;
}

// ── ADULT: Causal & Recourse ─────────────────────────────────────────────
async function loadAdultCausal(){
  const el = document.getElementById('adult-causal-content');
  const cfs = await api(API+'/api/counterfactuals');
  const d = await api(API+'/api/causal');
  if(!cfs || !d){el.innerHTML='<p style="color:#f04438">Backend not reachable.</p>';return;}

  const namesMap = {
    'age': 'Age',
    'education_num': 'Education Years',
    'capital_gain': 'Capital Gain',
    'capital_loss': 'Capital Loss',
    'hours_per_week': 'Hours Per Week'
  };

  function cfTable(rows){
    return rows.map((cf,i)=>`
      <div style="margin-bottom:18px;padding:18px;background:rgba(255,255,255,0.01);border-radius:12px;border:1px solid var(--border);">
        <div style="font-size:.82rem;color:var(--sg);margin-bottom:12px;font-weight:600;">Instance Profile #${i+1}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;font-size:.85rem;">
          <div style="background:rgba(248,113,113,0.03); padding:14px; border-radius:8px; border:1px solid rgba(248,113,113,0.08);">
            <div style="color:var(--c4);font-weight:700;margin-bottom:8px;font-size:.9rem;">Original Profile (Income: &le;$50K)</div>
            ${Object.entries(cf.original||{}).map(([k,v])=>`<div>${namesMap[k] || k}: <span style="color:var(--fg);font-weight:600;">${v}</span></div>`).join('')}
          </div>
          <div style="background:rgba(52,211,153,0.03); padding:14px; border-radius:8px; border:1px solid rgba(52,211,153,0.08);">
            <div style="color:var(--c3);font-weight:700;margin-bottom:8px;font-size:.9rem;">Counterfactual Recourse (Income: &gt;$50K)</div>
            ${Object.entries(cf.counterfactual||{}).map(([k,v])=>`<div>${namesMap[k] || k}: <span style="color:var(--c1);font-weight:600;">${v}</span></div>`).join('')}
          </div>
        </div>
      </div>`).join('');
  }

  el.innerHTML=`
  <div class="card">
    <h4>Causal Effect (Sex &rarr; Income &gt;50K)</h4>
    <div class="stat c2" style="margin-bottom:18px; max-width: 320px;"><div class="val">${d.adult.ate}</div><div class="lbl">Average Treatment Effect (ATE)</div></div>
    <p style="font-size:.88rem;color:var(--sg);line-height:1.7;margin:0;">${d.adult.interpretation}</p>
    <div id="ch-adult-causal-ate" class="chart-box" style="min-height:220px;margin-top:18px;"></div>
  </div>
  <div class="card">
    <h4>Actionable Recourse (DiCE Counterfactuals)</h4>
    ${cfTable(cfs.adult)}
  </div>`;

  const cfg={
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Outfit', color:'#F8FAFC'},
    margin:{t:30,b:30,l:40,r:20},
    xaxis:{gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}},
    yaxis:{gridcolor:'rgba(255,255,255,0.06)', linecolor:'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'}}
  };

  Plotly.newPlot('ch-adult-causal-ate',[
    {type:'bar',x:['Raw Disparity (DP-Diff)','Direct Causal Effect (ATE)'],y:[0.162,0.173],
      marker:{color:['#38BDF8','#34D399']},text:['0.162','0.173'],textposition:'outside'}],
    {...cfg,yaxis:{range:[0,.3]},title:{text:'Sex Effect: Observed vs Causal',font:{family:'Outfit',size:13,color:'#fff'}}});
}

// ── Predict COMPAS ────────────────────────────────────────────────────────
async function runPredict(){
  const body = {
    age: +document.getElementById('p-age').value,
    priors_count: +document.getElementById('p-priors').value,
    charge_degree_enc: +document.getElementById('p-charge').value,
    days_b_screening_arrest: +document.getElementById('p-days').value,
    length_of_stay: +document.getElementById('p-los').value,
  };
  
  document.getElementById('pred-idle-state').style.display = 'none';
  const r = document.getElementById('pred-result');
  r.style.display = 'block';
  document.getElementById('pred-label').innerHTML = '<div class="spinner"></div> Evaluating...';
  document.getElementById('pred-prob-badge').textContent = '';
  document.getElementById('ch-pred-contrib').innerHTML = '';
  
  const d = await fetch(API+'/api/predict/compas',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)
  }).then(r=>r.json()).catch(()=>null);
  
  if(!d||d.error){
    document.getElementById('pred-label').innerHTML=`<span style="color:var(--c4)">Error: ${d?.error||'Backend not reachable'}</span>`;
    return;
  }
  
  const isHigh = d.prediction===1;
  const pct = (d.probability*100).toFixed(1);
  document.getElementById('pred-label').innerHTML=
    `<div class="${isHigh?'risk-high':'risk-low'}">${isHigh?'High Bail/Detention Risk':'Low Bail/Detention Risk'}</div>`;
  document.getElementById('pred-prob-badge').textContent = `${pct}% Risk Probability`;
  
  const fill=document.getElementById('pred-bar-fill');
  fill.style.width=pct+'%';
  fill.style.background=isHigh?'var(--c4)':'var(--c3)';
  
  // Render local explanation chart
  if(d.contributions && d.contributions.length > 0){
    const namesMap = {
      'age': 'Age of Accused',
      'charge_degree_enc': 'Offense Type (Cognizable)',
      'priors_count': 'Past FIRs',
      'days_b_screening_arrest': 'Days from FIR to Arrest',
      'length_of_stay': 'Pre-trial Custody Duration (Days)'
    };
    
    // Sort contributions by magnitude
    const contribs = [...d.contributions].sort((a,b) => Math.abs(a.contribution) - Math.abs(b.contribution));
    const featLabels = contribs.map(c => namesMap[c.feature] || c.feature);
    const featVals = contribs.map(c => c.contribution);
    const featColors = featVals.map(v => v >= 0 ? 'rgba(248, 113, 113, 0.85)' : 'rgba(52, 211, 153, 0.85)');
    
    const trace = {
      type: 'bar',
      orientation: 'h',
      x: featVals,
      y: featLabels,
      marker: {
        color: featColors,
        line: { color: featVals.map(v => v >= 0 ? 'var(--c4)' : 'var(--c3)'), width: 1.5 }
      },
      text: featVals.map(v => (v >= 0 ? '+' : '') + v.toFixed(3)),
      textposition: 'outside'
    };
    
    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Outfit', color: '#F8FAFC', size: 11 },
      margin: { t: 10, b: 30, l: 140, r: 60 },
      xaxis: { title: 'Log-odds Contribution', gridcolor: 'rgba(255,255,255,0.06)', linecolor: 'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'} },
      yaxis: { gridcolor: 'rgba(255,255,255,0.06)', linecolor: 'rgba(255,255,255,0.1)', tickfont:{color:'#94A3B8'} }
    };
    
    Plotly.newPlot('ch-pred-contrib', [trace], layout, {responsive: true, displayModeBar: false});
  }
}

// ── Predict ADULT ────────────────────────────────────────────────────────
async function runAdultPredict(){
  const body = {
    age: +document.getElementById('p-a-age').value,
    education_num: +document.getElementById('p-a-education_num').value,
    capital_gain: +document.getElementById('p-a-capital_gain').value,
    capital_loss: +document.getElementById('p-a-capital_loss').value,
    hours_per_week: +document.getElementById('p-a-hours_per_week').value,
    workclass: document.getElementById('p-a-workclass').value,
    education: document.getElementById('p-a-education').value,
    marital_status: document.getElementById('p-a-marital_status').value,
    occupation: document.getElementById('p-a-occupation').value,
    relationship: document.getElementById('p-a-relationship').value,
    native_country: document.getElementById('p-a-native_country').value,
  };
  
  document.getElementById('a-pred-idle-state').style.display = 'none';
  const r = document.getElementById('a-pred-result');
  r.style.display = 'block';
  document.getElementById('a-pred-label').innerHTML = '<div class="spinner"></div> Evaluating...';
  document.getElementById('a-pred-prob-badge').textContent = '';
  
  const d = await fetch(API+'/api/predict/adult',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)
  }).then(r=>r.json()).catch(()=>null);
  
  if(!d||d.error){
    document.getElementById('a-pred-label').innerHTML=`<span style="color:var(--c4)">Error: ${d?.error||'Backend not reachable'}</span>`;
    return;
  }
  
  const isHigh = d.prediction===1;
  const pct = (d.probability*100).toFixed(1);
  document.getElementById('a-pred-label').innerHTML=
    `<div class="${isHigh?'risk-low':'risk-high'}" style="color:${isHigh?'var(--c3)':'var(--c4)'}">${isHigh?'&gt;$50K/year':'&le;$50K/year'}</div>`;
  document.getElementById('a-pred-prob-badge').textContent = `${pct}% Probability`;
  
  const fill=document.getElementById('a-pred-bar-fill');
  fill.style.width=pct+'%';
  fill.style.background=isHigh?'var(--c3)':'var(--c4)';
}

function loadReport(){
  document.getElementById('btn-download-pdf').href = `${API}/api/report`;
}

// Case Suggestions Logic
const caseSuggestions = {
  Theft: {
    text: "Theft offenses are often cognizable but carry moderate sentences. Suggestion: Age: 28, Past FIRs: 1, Offense Type: Cognizable, Delay: 0, Custody Stay: 3 days.",
    age: 28, priors: 1, charge: 1, days: 0, los: 3
  },
  Assault: {
    text: "Assault cases frequently arise from sudden disputes. Suggestion: Age: 34, Past FIRs: 0, Offense Type: Non-Cognizable, Delay: 1, Custody Stay: 1 day.",
    age: 34, priors: 0, charge: 0, days: 1, los: 1
  },
  Fraud: {
    text: "Financial frauds are serious, non-bailable, and heavily scrutinized. Suggestion: Age: 42, Past FIRs: 2, Offense Type: Cognizable, Delay: 3, Custody Stay: 14 days.",
    age: 42, priors: 2, charge: 1, days: 3, los: 14
  },
  Cyber: {
    text: "Cyber offenses are technical and usually cognizable. Suggestion: Age: 23, Past FIRs: 0, Offense Type: Cognizable, Delay: 0, Custody Stay: 2 days.",
    age: 23, priors: 0, charge: 1, days: 0, los: 2
  },
  Murder: {
    text: "Murder involves the highest level of gravity and pre-trial detention is standard. Suggestion: Age: 29, Past FIRs: 4, Offense Type: Cognizable, Delay: -2, Custody Stay: 120 days.",
    age: 29, priors: 4, charge: 1, days: -2, los: 120
  }
};

let activeSuggestionKey = 'Theft';

function onCaseTypeChange(val) {
  activeSuggestionKey = val;
  const box = document.getElementById('case-suggestions-text');
  if (box && caseSuggestions[val]) {
    box.textContent = caseSuggestions[val].text;
  }
}

function applyCaseSuggestion() {
  const sug = caseSuggestions[activeSuggestionKey];
  if (sug) {
    loadPreset(sug.age, sug.priors, sug.charge, sug.days, sug.los);
  }
}

// Chatbot UI handlers
async function sendCompasChat(msg) {
  if (!msg || !msg.trim()) return;
  const inputEl = document.getElementById('compas-chat-input');
  if (inputEl) inputEl.value = '';
  
  const chatBox = document.getElementById('compas-chat-box');
  chatBox.innerHTML += `
    <div style="margin-bottom:12px; font-size:.85rem; text-align:right;">
      <span class="chat-bubble-user">
        ${msg}
      </span>
    </div>
  `;
  chatBox.scrollTop = chatBox.scrollHeight;
  
  const tid = 'c-typing-' + Date.now();
  chatBox.innerHTML += `
    <div id="${tid}" style="margin-bottom:12px; font-size:.85rem; line-height:1.5;">
      <strong style="color:var(--c1);">NyayaAI:</strong> <span class="spinner" style="width:14px; height:14px; border-width:2px; vertical-align:middle;"></span> Thinking...
    </div>
  `;
  chatBox.scrollTop = chatBox.scrollHeight;
  
  const priorsVal = +document.getElementById('p-priors').value;
  const ageVal = +document.getElementById('p-age').value;
  const chargeVal = +document.getElementById('p-charge').value;
  const daysVal = +document.getElementById('p-days').value;
  const losVal = +document.getElementById('p-los').value;
  const caseType = document.getElementById('p-case-type').value;
  
  let predVal = 0;
  let probVal = 0.5;
  const labelEl = document.getElementById('pred-label');
  if (labelEl) {
    predVal = labelEl.textContent.includes('High') ? 1 : 0;
  }
  const probBadge = document.getElementById('pred-prob-badge');
  if (probBadge && probBadge.textContent) {
    probVal = parseFloat(probBadge.textContent) / 100 || 0.5;
  }
  
  const body = {
    message: msg,
    features: {
      age: ageVal,
      priors_count: priorsVal,
      charge_degree_enc: chargeVal,
      days_b_screening_arrest: daysVal,
      length_of_stay: losVal,
      case_type: caseType
    },
    prediction: predVal,
    probability: probVal
  };
  
  const res = await fetch(API + '/api/chatbot/compas', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(r => r.json()).catch(() => null);
  
  const typeEl = document.getElementById(tid);
  if (typeEl) {
    typeEl.remove();
  }
  
  const reply = res && res.response ? res.response : "I'm sorry, I couldn't reach NyayaAI. Please verify the backend is running.";
  const formattedReply = reply
    .replace(/\n/g, '<br/>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/### (.*?)(<br\/>|$)/g, '<h5 style="color:var(--c1); margin-top:8px;">$1</h5>')
    .replace(/- (.*?)(<br\/>|$)/g, '• $1$2');
    
  chatBox.innerHTML += `
    <div style="margin-bottom:16px; font-size:.85rem; line-height:1.5;">
      <strong style="color:var(--c1); display:flex; align-items:center; gap:6px;">⚖️ NyayaAI:</strong>
      <div class="chat-bubble-bot chat-bubble-bot-compas">${formattedReply}</div>
    </div>
  `;
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendAdultChat(msg) {
  if (!msg || !msg.trim()) return;
  const inputEl = document.getElementById('adult-chat-input');
  if (inputEl) inputEl.value = '';
  
  const chatBox = document.getElementById('adult-chat-box');
  chatBox.innerHTML += `
    <div style="margin-bottom:12px; font-size:.85rem; text-align:right;">
      <span class="chat-bubble-user" style="background: linear-gradient(135deg, #D97706 0%, #B45309 100%) !important; border-color: rgba(245, 158, 11, 0.3) !important; box-shadow: 0 4px 12px rgba(217, 119, 6, 0.2) !important;">
        ${msg}
      </span>
    </div>
  `;
  chatBox.scrollTop = chatBox.scrollHeight;
  
  const tid = 'a-typing-' + Date.now();
  chatBox.innerHTML += `
    <div id="${tid}" style="margin-bottom:12px; font-size:.85rem; line-height:1.5;">
      <strong style="color:var(--accent);">VikasAI:</strong> <span class="spinner" style="width:14px; height:14px; border-width:2px; vertical-align:middle; border-top-color:var(--accent);"></span> Thinking...
    </div>
  `;
  chatBox.scrollTop = chatBox.scrollHeight;
  
  const ageVal = +document.getElementById('p-a-age').value;
  const edNumVal = +document.getElementById('p-a-education_num').value;
  const capGainVal = +document.getElementById('p-a-capital_gain').value;
  const capLossVal = +document.getElementById('p-a-capital_loss').value;
  const hoursVal = +document.getElementById('p-a-hours_per_week').value;
  
  const workclassVal = document.getElementById('p-a-workclass').value;
  const educationVal = document.getElementById('p-a-education').value;
  const maritalVal = document.getElementById('p-a-marital_status').value;
  const occupationVal = document.getElementById('p-a-occupation').value;
  const relationshipVal = document.getElementById('p-a-relationship').value;
  
  let predVal = 0;
  let probVal = 0.5;
  const labelEl = document.getElementById('a-pred-label');
  if (labelEl) {
    predVal = labelEl.textContent.includes('>') ? 1 : 0;
  }
  const probBadge = document.getElementById('a-pred-prob-badge');
  if (probBadge && probBadge.textContent) {
    probVal = parseFloat(probBadge.textContent) / 100 || 0.5;
  }
  
  const body = {
    message: msg,
    features: {
      age: ageVal,
      education_num: edNumVal,
      capital_gain: capGainVal,
      capital_loss: capLossVal,
      hours_per_week: hoursVal,
      workclass: workclassVal,
      education: educationVal,
      marital_status: maritalVal,
      occupation: occupationVal,
      relationship: relationshipVal
    },
    prediction: predVal,
    probability: probVal
  };
  
  const res = await fetch(API + '/api/chatbot/adult', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(r => r.json()).catch(() => null);
  
  const typeEl = document.getElementById(tid);
  if (typeEl) {
    typeEl.remove();
  }
  
  const reply = res && res.response ? res.response : "I'm sorry, I couldn't reach VikasAI. Please verify the backend is running.";
  const formattedReply = reply
    .replace(/\n/g, '<br/>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/### (.*?)(<br\/>|$)/g, '<h5 style="color:var(--accent); margin-top:8px;">$1</h5>')
    .replace(/- (.*?)(<br\/>|$)/g, '• $1$2');
    
  chatBox.innerHTML += `
    <div style="margin-bottom:16px; font-size:.85rem; line-height:1.5;">
      <strong style="color:var(--accent); display:flex; align-items:center; gap:6px;">💼 VikasAI:</strong>
      <div class="chat-bubble-bot chat-bubble-bot-adult">${formattedReply}</div>
    </div>
  `;
  chatBox.scrollTop = chatBox.scrollHeight;
}

const loaders={
  overview: loadOverview,
  'compas-data': loadCompasData,
  'compas-predict': loadCompasPredict,
  'compas-fairness': loadCompasFairness,
  'compas-causal': loadCompasCausal,
  'adult-data': loadAdultData,
  'adult-predict': loadAdultPredict,
  'adult-fairness': loadAdultFairness,
  'adult-causal': loadAdultCausal,
  report: loadReport
};

// Load overview on startup
loadOverview();
onCaseTypeChange('Theft');
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML)

@app.get("/health")
def health():
    return {"status": "ok", "app": "FairXplain Frontend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)