
import re
import html

import streamlit as st
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "Shivendrasinh/legal-contract-risk-model"
LABELS = {0: "Low", 1: "Moderate", 2: "High"}
WEIGHT_LOW, WEIGHT_MOD, WEIGHT_HIGH = 1.0, 1.5, 3.0

st.set_page_config(
    page_title="AI-Based Legal Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
)

# Original charcoal-and-gold visual system
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --ink: #0a0a0f;
  --ink2: #14141e;
  --gold: #c9a84c;
  --cream: #f0ece2;
  --cream-dim: #b8b09a;
  --muted: #7d7d92;
  --border: rgba(201,168,76,.18);
}
.stApp {
  background: var(--ink);
  color: var(--cream);
  font-family: 'DM Sans', sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer {visibility: hidden;}
.block-container {
  max-width: 1100px;
  padding: 1.5rem 2rem 4rem;
}
.brand-nav {
  display:flex; justify-content:space-between; align-items:center;
  padding: 1rem 0 1.2rem; border-bottom:1px solid rgba(255,255,255,.07);
}
.brand {
  font-family:'Cormorant Garamond',serif;
  font-size:1.5rem; letter-spacing:.07em; color:var(--gold);
}
.brand span { color:var(--cream); }
.nav-tag {
  color:var(--muted); text-transform:uppercase;
  font-size:.68rem; letter-spacing:.16em;
}
.hero { max-width:850px; margin:0 auto; padding:4.5rem 0 2.5rem; }
.eyebrow {
  display:inline-block; color:var(--gold); border:1px solid var(--border);
  border-radius:30px; padding:.4rem .8rem; font-size:.7rem;
  letter-spacing:.16em; text-transform:uppercase; margin-bottom:1.5rem;
}
.hero h1 {
  font-family:'Cormorant Garamond',serif; font-size:clamp(2.8rem,6vw,4.5rem);
  font-weight:500; line-height:1.05; color:var(--cream); margin:0 0 1rem;
}
.hero h1 em { color:var(--gold); }
.hero p { color:var(--cream-dim); line-height:1.8; max-width:650px; }
.panel {
  background:var(--ink2); border:1px solid var(--border);
  border-radius:18px; overflow:hidden; padding:1.5rem;
}
.section-label {
  color:var(--gold); font-size:.7rem; letter-spacing:.15em;
  text-transform:uppercase; margin-bottom:.7rem;
}
.feature-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:1.5rem; }
.feature {
  background:var(--ink2); border:1px solid rgba(255,255,255,.07);
  border-radius:12px; padding:1.2rem;
}
.feature h4 { color:var(--cream); margin:.4rem 0; font-size:.95rem; }
.feature p { color:var(--muted); font-size:.8rem; line-height:1.6; margin:0; }
div[data-testid="stRadio"] > div { gap:.5rem; }
div[data-testid="stRadio"] label {
  background:#1c1c2a; border:1px solid var(--border);
  border-radius:9px; padding:.65rem 1rem;
}
div[data-testid="stTextArea"] textarea,
div[data-testid="stFileUploader"] section {
  background:#101019; color:var(--cream); border-color:var(--border);
}
div[data-testid="stTextArea"] textarea { min-height:220px; }
.stButton > button {
  background:var(--gold); color:#0a0a0f; border:0;
  border-radius:8px; font-weight:600; padding:.7rem 1.5rem;
}
.stButton > button:hover { background:#dab95a; color:#0a0a0f; }
.report-title {
  font-family:'Cormorant Garamond',serif; font-size:2.5rem;
  color:var(--cream); line-height:1.1;
}
.metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1.2rem 0; }
.metric {
  background:var(--ink2); border:1px solid rgba(255,255,255,.08);
  border-radius:14px; padding:1.3rem;
}
.metric-label { color:var(--muted); font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; }
.metric-value { font-family:'Cormorant Garamond',serif; font-size:2.5rem; margin:.3rem 0; }
.risk-high { color:#e06060; }
.risk-moderate { color:#d4974a; }
.risk-low { color:#5aac82; }
.clause {
  background:var(--ink2); border:1px solid rgba(255,255,255,.07);
  border-radius:12px; padding:1.2rem; margin:.8rem 0;
}
.clause-text { color:var(--cream-dim); line-height:1.7; margin:.8rem 0; }
.badge {
  display:inline-block; border:1px solid var(--border);
  border-radius:20px; padding:.3rem .7rem; color:var(--gold); font-size:.75rem;
}
@media(max-width:700px) {
  .feature-grid,.metrics { grid-template-columns:1fr; }
  .hero { padding:3rem 0 2rem; }
  .block-container { padding:1rem; }
  .brand { font-size:1.15rem; }
  .nav-tag { font-size:.55rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    model.to(torch.device("cpu"))
    return tokenizer, model


def clean_clause(clause):
    if clause is None:
        return ""
    text = clause.strip()
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", text)
    text = re.sub(r"^\s*\(\w+\)\s*", "", text)
    return text if len(text.split()) >= 4 else ""


def split_to_clauses(text):
    raw = re.split(r"(?<=[\.\;\:\?\!])\s+", text)
    return [c for c in (clean_clause(x) for x in raw) if c]


def analyze_text_with_model(text, tokenizer, model):
    clauses = split_to_clauses(text)
    if not clauses:
        return {"overall_score": 0.0, "risk_level": "Low", "high_risk_clauses": []}

    total_score, total_weights = 0.0, 0.0
    high_risk_clauses = []

    for clause in clauses:
        inputs = tokenizer(
            clause, truncation=True, padding=True, return_tensors="pt"
        )
        with torch.inference_mode():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        prob_high = float(probs[2])
        pred_class = int(probs.argmax())
        weight = {0: WEIGHT_LOW, 1: WEIGHT_MOD, 2: WEIGHT_HIGH}[pred_class]

        total_score += prob_high * weight
        total_weights += weight

        if pred_class == 2 or prob_high >= 0.7:
            high_risk_clauses.append({
                "clause": clause,
                "high_prob": round(prob_high * 100, 2),
                "pred": LABELS[pred_class],
            })

    score = round((total_score / total_weights) * 100, 2) if total_weights else 0.0
    level = "High" if score >= 65 else "Moderate" if score >= 35 else "Low"

    return {
        "overall_score": score,
        "risk_level": level,
        "high_risk_clauses": high_risk_clauses,
    }


# Header and hero matching the original page
st.markdown("""
<div class="brand-nav">
  <div class="brand">AI-Based <span>Risk Prediction</span></div>
  <div class="nav-tag">AI Contract Intelligence</div>
</div>
<div class="hero">
  <div class="eyebrow">● &nbsp; Powered by DistilBERT NLP</div>
  <h1>Analyze contract risk<br>with <em>precision</em></h1>
  <p>Upload any legal contract or paste its text. Our AI model identifies weighted risk scores and surfaces high-risk clauses in seconds.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Contract input</div>', unsafe_allow_html=True)

input_mode = st.radio(
    "Choose input method",
    ["Upload PDF", "Paste Text"],
    horizontal=True,
    label_visibility="collapsed",
    key="input_mode",
)

contract_text = ""
uploaded_pdf = None

if input_mode == "Upload PDF":
    uploaded_pdf = st.file_uploader(
        "Drop your PDF here or click to browse",
        type=["pdf"],
        key="contract_pdf",
    )
else:
    contract_text = st.text_area(
        "Paste the full contract text",
        height=230,
        placeholder="Paste the full contract text here. For best results, include complete clause sentences…",
        key="contract_text",
    )
    st.caption(f"{len(contract_text):,} characters")

st.caption("Best results with clean text PDFs. Scanned or image-based PDFs may require OCR.")
analyze_clicked = st.button("⌕  ANALYZE RISK", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
  <div class="feature"><div class="section-label">01 / Detection</div><h4>Clause-level detection</h4><p>Each sentence is scored independently to help surface clauses that need review.</p></div>
  <div class="feature"><div class="section-label">02 / Scoring</div><h4>Weighted risk scoring</h4><p>High-risk predictions carry greater weight in the overall risk score.</p></div>
  <div class="feature"><div class="section-label">03 / Documents</div><h4>PDF & text support</h4><p>Upload a text-based PDF or paste contract text for analysis.</p></div>
</div>
<p style="text-align:center;color:#7d7d92;font-size:.75rem;margin-top:2rem;">Automated screening only — not legal advice.</p>
""", unsafe_allow_html=True)


if analyze_clicked:
    text = contract_text.strip() if input_mode == "Paste Text" else ""

    if input_mode == "Upload PDF":
        if uploaded_pdf is None:
            st.error("Please upload a PDF contract.")
            st.stop()
        try:
            with pdfplumber.open(uploaded_pdf) as pdf:
                text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                ).strip()
        except Exception:
            st.error("Unable to read this PDF. Please try another file.")
            st.stop()

        if not text:
            st.error("No selectable text found. Scanned PDFs may require OCR.")
            st.stop()

    if not text:
        st.error("Please paste contract text or upload a PDF.")
        st.stop()

    try:
        with st.spinner("Loading DistilBERT and analyzing clauses..."):
            tokenizer, model = load_model()
            result = analyze_text_with_model(text, tokenizer, model)
    except Exception as exc:
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
        st.stop()

    score = result["overall_score"]
    level = result["risk_level"]
    css_level = level.lower()
    findings = result["high_risk_clauses"]

    st.markdown("---")
    st.markdown("""
    <div class="section-label">Risk intelligence report</div>
    <div class="report-title">Contract Analysis<br>Complete</div>
    <p style="color:#7d7d92">Analyzed with DistilBERT NLP</p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics">
      <div class="metric"><div class="metric-label">Overall Risk Score</div>
        <div class="metric-value risk-{css_level}">{score:.2f}%</div>
        <div class="badge">{level} Risk</div></div>
      <div class="metric"><div class="metric-label">Risk Level</div>
        <div class="metric-value risk-{css_level}">{level}</div>
        <div style="color:#7d7d92;font-size:.8rem">{"Score ≥ 65%" if level == "High" else "Score 35–64%" if level == "Moderate" else "Score < 35%"}</div></div>
      <div class="metric"><div class="metric-label">High-Risk Clauses</div>
        <div class="metric-value risk-high">{len(findings)}</div>
        <div style="color:#7d7d92;font-size:.8rem">Flagged for review</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">Risk exposure meter · {score:.2f}%</div>', unsafe_allow_html=True)
    st.progress(min(max(score / 100, 0.0), 1.0))

    st.markdown(f"""
    <div class="panel">
      <div class="section-label">High-risk clauses · {len(findings)} flagged</div>
    """, unsafe_allow_html=True)

    if findings:
        for i, item in enumerate(findings, 1):
            safe_clause = html.escape(item["clause"])
            safe_pred = html.escape(item["pred"])
            st.markdown(f"""
            <div class="clause">
              <div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap">
                <span style="color:#7d7d92;font-size:.7rem;letter-spacing:.12em">CLAUSE {i}</span>
                <span class="risk-high">{item["high_prob"]:.2f}% risk probability</span>
              </div>
              <div class="clause-text">{safe_clause}</div>
              <span class="badge">{safe_pred} Risk</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No high-risk clauses met the reporting threshold.")

    st.markdown("</div>", unsafe_allow_html=True)
