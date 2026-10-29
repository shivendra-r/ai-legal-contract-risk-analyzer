
import re
from functools import lru_cache

import streamlit as st
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "Shivendrasinh/legal-contract-risk-model"

LABELS = {0: "Low", 1: "Moderate", 2: "High"}
WEIGHT_LOW = 1.0
WEIGHT_MOD = 1.5
WEIGHT_HIGH = 3.0

st.set_page_config(
    page_title="AI Legal Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    model.to(torch.device("cpu"))
    return tokenizer, model


def clean_clause(clause: str) -> str:
    if clause is None:
        return ""

    text = clause.strip()
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", text)
    text = re.sub(r"^\s*\(\w+\)\s*", "", text)

    if len(text.split()) < 4:
        return ""

    return text


def split_to_clauses(text: str):
    raw = re.split(r"(?<=[\.\;\:\?\!])\s+", text)
    clauses = [clean_clause(item) for item in raw]
    return [clause for clause in clauses if clause]


def analyze_text_with_model(text: str, tokenizer, model):
    clauses = split_to_clauses(text)

    if not clauses:
        return {
            "overall_score": 0.0,
            "risk_level": "Low",
            "high_risk_clauses": [],
        }

    total_weighted_score = 0.0
    total_weights = 0.0
    high_risk_clauses = []

    for clause in clauses:
        inputs = tokenizer(
            clause,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = model(**inputs)
            probs = torch.softmax(
                outputs.logits, dim=-1
            ).cpu().numpy()[0]

        prob_low = float(probs[0])
        prob_mod = float(probs[1])
        prob_high = float(probs[2])

        pred_class = int(probs.argmax())

        if pred_class == 2:
            weight = WEIGHT_HIGH
        elif pred_class == 1:
            weight = WEIGHT_MOD
        else:
            weight = WEIGHT_LOW

        total_weighted_score += prob_high * weight
        total_weights += weight

        if pred_class == 2 or prob_high >= 0.7:
            high_risk_clauses.append({
                "clause": clause,
                "high_prob": round(prob_high * 100, 2),
                "pred": LABELS[pred_class],
            })

    average = (
        total_weighted_score / total_weights
        if total_weights > 0
        else 0.0
    )

    overall_score = round(average * 100, 2)

    if overall_score >= 65:
        risk_level = "High"
    elif overall_score >= 35:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "overall_score": overall_score,
        "risk_level": risk_level,
        "high_risk_clauses": high_risk_clauses,
    }


def extract_pdf_text(uploaded_file):
    pages = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)

    return "\n".join(pages)


st.title("⚖️ AI Legal Contract Risk Analyzer")
st.write(
    "Analyze contract clauses and identify potential risk levels "
    "using your trained DistilBERT classification model."
)

st.info(
    "This tool provides automated risk screening, not legal advice. "
    "Review important findings with a qualified legal professional."
)

contract_text = st.text_area(
    "Paste contract text",
    height=250,
    placeholder="Paste your contract text here...",
)

uploaded_pdf = st.file_uploader(
    "Or upload a text-based PDF contract",
    type=["pdf"],
)

if st.button("Analyze Contract", type="primary"):
    text = contract_text.strip()

    if uploaded_pdf is not None:
        try:
            pdf_text = extract_pdf_text(uploaded_pdf)
            if pdf_text.strip():
                text = pdf_text.strip()
            elif not text:
                st.error(
                    "No selectable text was found in the PDF. "
                    "Scanned PDFs require OCR."
                )
                st.stop()
        except Exception:
            st.error("Unable to read this PDF. Please try another file.")
            st.stop()

    if not text:
        st.warning("Please paste contract text or upload a PDF.")
        st.stop()

    try:
        with st.spinner("Loading model and analyzing contract..."):
            tokenizer, model = load_model()
            result = analyze_text_with_model(text, tokenizer, model)

        st.subheader("Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Overall Risk Score", f"{result['overall_score']}%")

        with col2:
            st.metric("Overall Risk Level", result["risk_level"])

        st.progress(
            min(max(result["overall_score"] / 100, 0.0), 1.0)
        )

        st.subheader("High-Risk Clauses")

        findings = result["high_risk_clauses"]

        if findings:
            for index, finding in enumerate(findings, start=1):
                with st.expander(
                    f"Clause {index} — {finding['pred']} Risk "
                    f"({finding['high_prob']}% high-risk probability)"
                ):
                    st.write(finding["clause"])
        else:
            st.success("No clauses met the high-risk reporting threshold.")

    except Exception as exc:
        st.error(
            "Analysis failed. Please try again later. "
            f"Technical detail: {type(exc).__name__}: {exc}"
        )
