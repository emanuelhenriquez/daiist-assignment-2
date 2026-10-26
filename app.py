"""
Assignment 2 - Gradio dashboard: chest X-ray pneumonia triage.

Run with:  uv run python main.py app     (after main.py train)

Never trains anything. It loads the committed model in weights/best.pt and the
results train.py saved in results/ (test predictions, metrics, cost sweep).

Tabs:
  1. Try an X-ray     - upload an image, see the probabilities and the decision
  2. Thresholds & cost - two sliders (pass / review / fail), decision matrix,
                         cost per 1,000 X-rays, share sent to a radiologist,
                         staff needed, comparison with the alternatives
  3. Cost curve        - expected cost against a single threshold, choice marked
  4. Per-class results - precision / recall per class, confusion matrix, baselines
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costs

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights" / "best.pt"
if not (RESULTS / "metrics.json").exists() or not WEIGHTS.exists():
    raise SystemExit("Missing weights/best.pt or results/. Run `uv run python main.py train` first.")

metrics = json.loads((RESULTS / "metrics.json").read_text())
test = pd.read_csv(RESULTS / "predictions_test.csv")
sweep = pd.read_csv(RESULTS / "cost_sweep.csv")
model = YOLO(str(WEIGHTS))
CLASSES = metrics["classes"]
NORMAL = metrics["normal_class"]
C = metrics["costs"]
A = C["assumptions"]
Y_POS = test["is_pneumonia"].to_numpy()
SCORE = test["score"].to_numpy()


# ---------------------------------------------------------------------------
# Tab 1: upload an X-ray
# ---------------------------------------------------------------------------
def decide(score, low, high):
    if score < low:
        return "PASS: looks normal"
    if score >= high:
        return "FAIL: pneumonia, start treatment"
    return "REVIEW: send to a radiologist"


def predict_image(image, low, high):
    if image is None:
        return None, "Upload a chest X-ray."
    r = model.predict(image, imgsz=metrics["training"]["imgsz"], verbose=False)[0]
    probs = r.probs.data.cpu().numpy()
    probs_d = {model.names[i]: float(p) for i, p in enumerate(probs)}
    score = 1 - probs_d[NORMAL]
    text = (f"### {decide(score, low, high)}\n"
            f"Pneumonia score = 1 − P({NORMAL}) = **{score:.2f}**  "
            f"(pass below {low:.2f}, fail from {high:.2f})")
    return probs_d, text


# ---------------------------------------------------------------------------
# Tab 2: thresholds, decision matrix, cost
# ---------------------------------------------------------------------------
def threshold_view(low, high, c_fn, c_fp, c_rev, base_rate, cases, minutes, hours):
    high = max(high, low)
    kw = dict(base_rate=base_rate, c_fn=c_fn, c_fp=c_fp, c_review=c_rev)
    r = costs.rates(Y_POS, SCORE, low, high)
    cost = costs.expected_cost(r, **kw)
    share = costs.review_share(r, base_rate)
    staff = costs.staff_needed(share, cases, minutes, hours)

    # Decision matrix on the test split (counts) - rows: truth, cols: decision
    pos, neg = Y_POS == 1, Y_POS == 0
    dec = np.where(SCORE < low, "Pass", np.where(SCORE >= high, "Fail", "Review"))
    mat = [[int(((dec == d) & m).sum()) for d in ["Pass", "Review", "Fail"]] for m in [neg, pos]]
    labels = [["correct", "review", "FALSE ALARM"], ["MISSED", "review", "correct"]]
    fig = go.Figure(go.Heatmap(
        z=mat, x=["Pass (normal)", "Review (radiologist)", "Fail (pneumonia)"],
        y=["Actually normal", "Actually pneumonia"],
        text=[[f"{labels[i][j]}<br><b>{mat[i][j]}</b>" for j in range(3)] for i in range(2)],
        texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title=f"Test split ({len(Y_POS)} X-rays): truth vs decision",
                      yaxis_autorange="reversed", height=330, margin=dict(t=50, b=20))

    alone_t = C["single_threshold"]
    alone = 1000 * costs.expected_cost(costs.rates(Y_POS, SCORE, alone_t, alone_t), **kw)
    md = (
        f"### Expected cost: **€{1000 * cost:,.0f} per 1,000 X-rays**\n"
        f"- Missed pneumonia: {r['pos_pass']:.1%} of sick patients (× €{c_fn:,.0f})\n"
        f"- False alarms: {r['neg_fail']:.1%} of healthy patients (× €{c_fp:,.0f})\n"
        f"- **Sent to a radiologist: {share:.1%}** of X-rays (× €{c_rev:,.0f})\n\n"
        f"**Workload:** {staff['reviews_per_day']:.0f} reviews/day × {minutes:.0f} min = "
        f"{staff['review_hours_per_day']:.1f} h/day → **{staff['radiologists_needed']:.2f} "
        f"radiologists** at {hours:.0f} h/day each.\n\n"
        f"**Compare (per 1,000 X-rays, same assumptions):**\n"
        f"- Do nothing (everyone passes): €{1000 * base_rate * c_fn:,.0f}\n"
        f"- Radiologist reads everything: €{1000 * c_rev:,.0f}\n"
        f"- Model alone at {alone_t:.2f}: €{alone:,.0f}\n"
        f"- This setting: €{1000 * cost:,.0f}\n\n"
        f"*Rates are measured on the test split, then re-weighted to a base rate of "
        f"{base_rate:.0%} pneumonia (the dataset itself is ~74% pneumonia).*")
    return fig, md


# ---------------------------------------------------------------------------
# Tab 3: cost curve
# ---------------------------------------------------------------------------
def cost_curve():
    t = C["single_threshold"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep.threshold, y=1000 * sweep.cost_val, name="validation (used to choose)",
                             line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=sweep.threshold, y=1000 * sweep.cost_test, name="test (reported)",
                             line=dict(color="#ff7f0e", dash="dot")))
    fig.add_vline(x=t, line_dash="dash", annotation_text=f"chosen {t:.2f}")
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", annotation_text="0.5",
                  annotation_position="bottom right")
    fig.update_layout(title="Model alone: expected cost per 1,000 X-rays vs threshold",
                      xaxis_title="Threshold on pneumonia score", yaxis_title="€ per 1,000 X-rays",
                      height=420)
    return fig


def strategies_table():
    rows = []
    for name, s in C["strategies"].items():
        rows.append({"Strategy": name, "€ per 1,000 X-rays": f"{s['cost_per_1000']:,.0f}",
                     "Sent to radiologist": f"{s['review_share']:.1%}",
                     "Radiologists needed": s.get("radiologists_needed", 0.0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tab 4: per-class results
# ---------------------------------------------------------------------------
def overall_table():
    rows = []
    for name, m in metrics["models"].items():
        b = m["binary_at_argmax"]
        rows.append({"Model": name, "Accuracy (3 classes)": m["accuracy"],
                     "Pneumonia sensitivity": b["sensitivity"],
                     "Specificity (normal kept normal)": b["specificity"]})
    return pd.DataFrame(rows)


def per_class(model_name):
    m = metrics["models"][model_name]
    table = pd.DataFrame(m["per_class"])
    cm = np.array(m["confusion_matrix"])
    fig = go.Figure(go.Heatmap(z=cm, x=CLASSES, y=CLASSES, text=cm, texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(title=f"Confusion matrix (test): {model_name}", xaxis_title="Predicted",
                      yaxis_title="Actual", yaxis_autorange="reversed", height=400)
    return table, fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
LOW, HIGH = C["low"], C["high"]
with gr.Blocks(title="Pneumonia triage") as demo:
    gr.Markdown(
        "# Chest X-ray pneumonia triage\n"
        f"Finetuned yolo11n-cls on Roboflow `{metrics['training']['dataset']}`. "
        "Pneumonia (bacterial or viral) is the **positive** class. "
        "Loaded from `weights/best.pt` and `results/`; nothing is retrained.")
    with gr.Row():
        low = gr.Slider(0, 1, value=LOW, step=0.01, label="Pass below (low threshold)")
        high = gr.Slider(0, 1, value=HIGH, step=0.01, label="Fail from (high threshold)")

    with gr.Tab("1 · Try an X-ray"):
        with gr.Row():
            img = gr.Image(type="pil", label="Chest X-ray")
            with gr.Column():
                probs = gr.Label(label="Class probabilities")
                verdict = gr.Markdown()
        for comp in [img, low, high]:
            comp.change(predict_image, [img, low, high], [probs, verdict])

    with gr.Tab("2 · Thresholds & cost"):
        with gr.Accordion("Assumptions (edit to test sensitivity)", open=False):
            with gr.Row():
                c_fn = gr.Number(A["COST_FN"], label="€ missed pneumonia (FN)")
                c_fp = gr.Number(A["COST_FP"], label="€ false alarm (FP)")
                c_rev = gr.Number(A["COST_REVIEW"], label="€ radiologist review")
                base = gr.Slider(0.01, 0.9, value=A["BASE_RATE"], step=0.01, label="Base rate (share with pneumonia)")
            with gr.Row():
                cases = gr.Number(A["CASES_PER_DAY"], label="X-rays per day")
                minutes = gr.Number(A["MINUTES_PER_REVIEW"], label="Minutes per review")
                hours = gr.Number(A["REVIEW_HOURS_PER_DAY"], label="Review hours per radiologist per day")
        with gr.Row():
            matrix = gr.Plot()
            summary = gr.Markdown()
        inputs = [low, high, c_fn, c_fp, c_rev, base, cases, minutes, hours]
        for comp in inputs:
            comp.change(threshold_view, inputs, [matrix, summary])
        demo.load(threshold_view, inputs, [matrix, summary])

    with gr.Tab("3 · Cost curve"):
        gr.Plot(cost_curve())
        gr.Markdown(f"**Chosen on validation:** model alone at **{C['single_threshold']:.2f}**; "
                    f"with review: pass below **{LOW:.2f}**, review between, fail from **{HIGH:.2f}**. "
                    "Strategies on the test split, with the default assumptions:")
        gr.Dataframe(strategies_table(), interactive=False)

    with gr.Tab("4 · Per-class results"):
        gr.Dataframe(overall_table(), interactive=False, label="Test split, decision = most likely class")
        pick = gr.Dropdown(list(metrics["models"]), value=list(metrics["models"])[0], label="Model")
        with gr.Row():
            pc_table = gr.Dataframe(interactive=False, label="Per-class precision / recall")
            pc_fig = gr.Plot()
        pick.change(per_class, pick, [pc_table, pc_fig])
        demo.load(per_class, pick, [pc_table, pc_fig])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
