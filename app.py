# app.py
import re
from flask import Flask, render_template, request
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

MODEL_ID = "Shivendrasinh/legal-contract-risk-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


LABELS = {0: "Low", 1: "Moderate", 2: "High"}

WEIGHT_LOW = 1.0
WEIGHT_MOD = 1.5
WEIGHT_HIGH = 3.0


def clean_clause(clause: str) -> str:
    """Remove leading numbering and trim; return empty string if too short."""
    if clause is None:
        return ""
    s = clause.strip()

    s = re.sub(r'^\s*\d+[\.\)]\s*', '', s)
    s = re.sub(r'^\s*\(\w+\)\s*', '', s)

    if len(s.split()) < 4:
        return ""
    return s

def split_to_clauses(text: str):
 
    raw = re.split(r'(?<=[\.\;\:\?\!])\s+', text)
    clauses = [clean_clause(r) for r in raw]
    return [c for c in clauses if c]


def analyze_text_with_model(text: str):
    clauses = split_to_clauses(text)
    if not clauses:
        return {
            "overall_score": 0.0,
            "risk_level": "Low",
            "high_risk_clauses": []
        }

    total_weighted_score = 0.0
    total_weights = 0.0
    high_risk_clauses = []

    for clause in clauses:
        
        inputs = tokenizer(clause, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]  # [low,mod,high]
        prob_low, prob_mod, prob_high = float(probs[0]), float(probs[1]), float(probs[2])

        
        pred_class = int(probs.argmax())

        if pred_class == 2:
            w = WEIGHT_HIGH
        elif pred_class == 1:
            w = WEIGHT_MOD
        else:
            w = WEIGHT_LOW

        contribution = prob_high * w 
        total_weighted_score += contribution
        total_weights += w

        
        if pred_class == 2 or prob_high >= 0.7:
            high_risk_clauses.append({
                "clause": clause,
                "high_prob": round(prob_high * 100, 2),
                "pred": LABELS[pred_class]
            })

    # Normalize overall score to percentage (0-100)
    # avg weighted high-prob = total_weighted_score / total_weights (in 0..1)
    avg_weighted_high_prob = (total_weighted_score / total_weights) if total_weights > 0 else 0.0
    overall_score = round(avg_weighted_high_prob * 100, 2)

    # derive risk level thresholds
    if overall_score >= 65:
        risk_level = "High"
    elif overall_score >= 35:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "overall_score": overall_score,
        "risk_level": risk_level,
        "high_risk_clauses": high_risk_clauses
    }

# routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text_input = request.form.get("contract_text", "").strip()
    file = request.files.get("file")

    # accept pasted text
    if text_input:
        text = text_input

    # accept PDF upload without saving to disk
    elif file and file.filename.lower().endswith(".pdf"):
        all_text = []
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    all_text.append(t)
        text = "\n".join(all_text)

    else:
        return render_template("index.html", error="Please paste contract text or upload a PDF.")

    result = analyze_text_with_model(text)
    # result contains overall_score (float), risk_level (str), high_risk_clauses (list)
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
