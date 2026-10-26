"""
Assignment 2 - training pipeline: chest X-ray pneumonia triage.

Run with:  uv run python main.py train

Steps (no manual work in between):
  1. Download the dataset from Roboflow Universe (key via api_key.py).
  2. Split it 70/15/15 into train / val / test, keeping all X-rays of the same
     patient in the same split (the published version has no split at all).
  3. Describe the dataset: images per class per split, image size.
  4. Finetune yolo11n-cls from its pretrained checkpoint.
  5. Evaluate on the test split against two baselines:
       - trivial: always predict the majority class
       - pretrained yolo11n-cls zero-shot (ImageNet classes, no finetuning)
  6. Per-class precision / recall and the confusion matrix.
  7. Cost sweep: one threshold (model alone) and two thresholds
     (pass / radiologist review / fail). Chosen on VALIDATION, reported on TEST.
  8. Save everything app.py needs into results/, and the model into weights/.
"""

import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import costs

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "split"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"

# --- Dataset (Roboflow Universe) ---------------------------------------------
WORKSPACE = "inatel"
PROJECT = "pneumonia-classification-imrcv"
VERSION = 1

# --- Training settings --------------------------------------------------------
BASE_MODEL = "yolo11n-cls.pt"
IMGSZ = 224
EPOCHS = int(os.environ.get("EPOCHS", 15))
BATCH = 32
SEED = 0
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download() -> Path:
    """Return a folder whose sub-folders are class names full of images."""
    override = os.environ.get("RAW_DATA_DIR")  # only used for offline testing
    if override:
        return Path(override)
    if not RAW_DIR.exists():
        from roboflow import Roboflow
        from api_key import load_api_key

        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "folder", location=str(RAW_DIR))
    # The published version puts everything in train/; accept any split folders.
    return RAW_DIR


def collect_images(raw: Path) -> pd.DataFrame:
    rows = []
    for img in raw.rglob("*"):
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            rows.append({"path": img, "label": img.parent.name})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No images found under {raw}")
    return df


def patient_id(path: Path) -> str:
    """Group X-rays of the same patient so they never span two splits.

    Roboflow keeps the original file name in front of '_jpeg.rf.<hash>':
    pneumonia files look like 'person1001_bacteria_2932', normal ones like
    'IM-0656-0001-0001' or 'NORMAL2-IM-0555-0001-0002'.
    """
    name = path.name.split(".rf.")[0]
    m = re.match(r"(person\d+)_", name)
    if m:
        return m.group(1)
    m = re.match(r"((?:NORMAL2-)?IM-\d+)", name)
    if m:
        return m.group(1)
    return name  # unknown pattern: treat the image as its own patient


# ---------------------------------------------------------------------------
# 2. Patient-grouped, stratified split
# ---------------------------------------------------------------------------
def make_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["patient"] = df["path"].map(patient_id)
    # Each patient goes wholly into one split; patients are shuffled within
    # their (majority) label so every split keeps the same class mix.
    patient_label = df.groupby("patient")["label"].agg(lambda s: s.value_counts().index[0])
    rng = random.Random(SEED)
    assign = {}
    for label, patients in patient_label.groupby(patient_label):
        ids = sorted(patients.index)
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = round(n * SPLIT["train"]), round(n * SPLIT["val"])
        for i, p in enumerate(ids):
            assign[p] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    df["split"] = df["patient"].map(assign)

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    for row in df.itertuples():
        dest = SPLIT_DIR / row.split / row.label / row.path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(row.path, dest)        # hard link: no extra disk space
        except OSError:
            shutil.copy2(row.path, dest)
        df.loc[row.Index, "split_path"] = str(dest)
    return df


# ---------------------------------------------------------------------------
# 3. Describe the dataset
# ---------------------------------------------------------------------------
def describe(df: pd.DataFrame) -> dict:
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    sizes = Counter()
    for p in df["path"].sample(min(200, len(df)), random_state=SEED):
        with Image.open(p) as im:
            sizes[f"{im.width}x{im.height}"] += 1
    stats = {
        "images_total": int(len(df)),
        "patients_total": int(df["patient"].nunique()),
        "per_split_per_class": {s: {k: int(v) for k, v in row.items()}
                                for s, row in counts.iterrows()},
        "per_class_total": {k: int(v) for k, v in df["label"].value_counts().items()},
        "image_sizes_sample": dict(sizes.most_common(5)),
        "patients_with_several_images": int((df.groupby("patient").size() > 1).sum()),
    }
    print("\nImages per split and class:\n" + counts.to_string())
    return stats


# ---------------------------------------------------------------------------
# 4. Finetune
# ---------------------------------------------------------------------------
def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def finetune() -> Path:
    from ultralytics import YOLO

    device = pick_device()
    print(f"\nTraining {BASE_MODEL} for {EPOCHS} epochs on {device} ...")
    model = YOLO(BASE_MODEL)
    model.train(data=str(SPLIT_DIR), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                seed=SEED, deterministic=True, device=device, workers=2,
                project=str(RUNS), name="pneumonia", exist_ok=True, plots=False)
    best = RUNS / "pneumonia" / "weights" / "best.pt"
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy2(best, WEIGHTS / "best.pt")
    return WEIGHTS / "best.pt"


# ---------------------------------------------------------------------------
# 5-6. Predict and evaluate
# ---------------------------------------------------------------------------
def predict_probs(model, paths: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], imgsz=IMGSZ, verbose=False):
            out.append(r.probs.data.cpu().numpy())
    return np.vstack(out)


def per_class_report(y_true, y_pred, classes) -> tuple[pd.DataFrame, list]:
    cm = pd.crosstab(pd.Categorical(y_true, classes), pd.Categorical(y_pred, classes),
                     dropna=False).reindex(index=classes, columns=classes, fill_value=0)
    rows = []
    for c in classes:
        tp = cm.loc[c, c]
        prec = tp / cm[c].sum() if cm[c].sum() else 0.0
        rec = tp / cm.loc[c].sum() if cm.loc[c].sum() else 0.0
        rows.append({"class": c, "support": int(cm.loc[c].sum()),
                     "precision": round(float(prec), 3), "recall": round(float(rec), 3)})
    return pd.DataFrame(rows), cm.values.tolist()


def binary_summary(y_true, y_pred, normal) -> dict:
    pos_true = np.array([y != normal for y in y_true])
    pos_pred = np.array([y != normal for y in y_pred])
    tp = int((pos_true & pos_pred).sum()); fn = int((pos_true & ~pos_pred).sum())
    fp = int((~pos_true & pos_pred).sum()); tn = int((~pos_true & ~pos_pred).sum())
    return {"sensitivity": round(tp / max(tp + fn, 1), 3),
            "specificity": round(tn / max(tn + fp, 1), 3),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def zero_shot_baseline(df: pd.DataFrame, classes) -> list:
    """Pretrained yolo11n-cls knows 1,000 ImageNet classes and none of ours.

    Without any training of weights, we map each ImageNet class it outputs to
    the label that most often carried that ImageNet class in the TRAIN split,
    then apply that lookup to the test split. Unseen ImageNet classes fall
    back to the majority label.
    """
    from ultralytics import YOLO

    zs = YOLO(BASE_MODEL)
    tr = df[df.split == "train"]
    te = df[df.split == "test"]
    tr_top = predict_probs(zs, tr["split_path"].tolist()).argmax(1)
    te_top = predict_probs(zs, te["split_path"].tolist()).argmax(1)
    votes = defaultdict(Counter)
    for k, lab in zip(tr_top, tr["label"]):
        votes[k][lab] += 1
    majority = tr["label"].value_counts().index[0]
    lookup = {k: v.most_common(1)[0][0] for k, v in votes.items()}
    names = zs.names
    top_classes = Counter(names[k] for k in te_top).most_common(5)
    print(f"Zero-shot model's most common ImageNet guesses on test: {top_classes}")
    return [lookup.get(k, majority) for k in te_top], top_classes


def evaluate(df: pd.DataFrame, weights: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    classes = [model.names[i] for i in range(len(model.names))]
    normal = next(c for c in classes if "normal" in c.lower())
    normal_idx = classes.index(normal)

    preds = {}
    for split in ["val", "test"]:
        part = df[df.split == split].reset_index(drop=True)
        probs = predict_probs(model, part["split_path"].tolist())
        out = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
        out.insert(0, "label", part["label"])
        out.insert(0, "path", part["split_path"].map(lambda p: str(Path(p).relative_to(ROOT))))
        out["pred"] = [classes[i] for i in probs.argmax(1)]
        out["score"] = 1 - probs[:, normal_idx]   # pneumonia score
        out["is_pneumonia"] = (out["label"] != normal).astype(int)
        out.to_csv(RESULTS / f"predictions_{split}.csv", index=False)
        preds[split] = out

    test = preds["test"]
    y_true = test["label"].tolist()
    results = {"classes": classes, "positive_class": "pneumonia (bacterial or viral)",
               "normal_class": normal, "models": {}}

    # Finetuned model
    table, cm = per_class_report(y_true, test["pred"].tolist(), classes)
    results["models"]["finetuned yolo11n-cls"] = {
        "accuracy": round(float((test["pred"] == test["label"]).mean()), 3),
        "per_class": table.to_dict("records"), "confusion_matrix": cm,
        "binary_at_argmax": binary_summary(y_true, test["pred"].tolist(), normal)}

    # Trivial baseline: majority class of the training split
    majority = df[df.split == "train"]["label"].value_counts().index[0]
    maj_pred = [majority] * len(y_true)
    table, cm = per_class_report(y_true, maj_pred, classes)
    results["models"][f"trivial: always '{majority}'"] = {
        "accuracy": round(float(np.mean([y == majority for y in y_true])), 3),
        "per_class": table.to_dict("records"), "confusion_matrix": cm,
        "binary_at_argmax": binary_summary(y_true, maj_pred, normal)}

    # Zero-shot pretrained model
    zs_pred, zs_top = zero_shot_baseline(df, classes)
    table, cm = per_class_report(y_true, zs_pred, classes)
    results["models"]["zero-shot pretrained yolo11n-cls (ImageNet)"] = {
        "accuracy": round(float(np.mean([a == b for a, b in zip(y_true, zs_pred)])), 3),
        "per_class": table.to_dict("records"), "confusion_matrix": cm,
        "binary_at_argmax": binary_summary(y_true, zs_pred, normal),
        "most_common_imagenet_guesses": zs_top}
    return results, preds


# ---------------------------------------------------------------------------
# 7. Costs and thresholds (choose on val, report on test)
# ---------------------------------------------------------------------------
def cost_analysis(preds: dict) -> dict:
    val, test = preds["val"], preds["test"]

    sweep_val = costs.single_threshold_sweep(val["is_pneumonia"], val["score"])
    sweep_test = costs.single_threshold_sweep(test["is_pneumonia"], test["score"])
    sweep = sweep_val.rename(columns={"cost_per_case": "cost_val"})[["threshold", "cost_val"]]
    sweep["cost_test"] = sweep_test["cost_per_case"]
    sweep["fn_rate_test"] = sweep_test["fn_rate"]
    sweep["fp_rate_test"] = sweep_test["fp_rate"]
    sweep.to_csv(RESULTS / "cost_sweep.csv", index=False)
    t_star = float(sweep_val.loc[sweep_val["cost_per_case"].idxmin(), "threshold"])

    grid = costs.two_threshold_search(val["is_pneumonia"], val["score"])
    best = grid.loc[grid["cost_per_case"].idxmin()]
    low, high = float(best["low"]), float(best["high"])

    def on_test(lo, hi):
        r = costs.rates(test["is_pneumonia"], test["score"], lo, hi)
        share = costs.review_share(r)
        return {"cost_per_1000": round(1000 * costs.expected_cost(r), 0),
                "review_share": round(share, 3),
                "missed_pneumonia_rate": round(r["pos_pass"], 3),
                "false_alarm_rate": round(r["neg_fail"], 3),
                **{k: round(v, 2) for k, v in costs.staff_needed(share).items()}}

    br = costs.BASE_RATE
    strategies = {
        "do nothing (everyone passes)": {
            "cost_per_1000": round(1000 * br * costs.COST_FN, 0), "review_share": 0.0},
        "radiologist reads everything": {
            "cost_per_1000": round(1000 * costs.COST_REVIEW, 0), "review_share": 1.0,
            **{k: round(v, 2) for k, v in costs.staff_needed(1.0).items()}},
        f"model alone (threshold {t_star:.2f})": on_test(t_star, t_star),
        f"model + review ({low:.2f} to {high:.2f})": on_test(low, high),
    }
    print("\nStrategies on TEST (per 1,000 X-rays):")
    for k, v in strategies.items():
        print(f"  {k:40s} €{v['cost_per_1000']:>9,.0f}   review {v['review_share']:.1%}")
    return {"single_threshold": t_star, "low": low, "high": high,
            "strategies": strategies,
            "assumptions": {k: getattr(costs, k) for k in [
                "COST_FN", "COST_FP", "COST_REVIEW", "COST_TP", "COST_TN", "BASE_RATE",
                "CASES_PER_DAY", "MINUTES_PER_REVIEW", "REVIEW_HOURS_PER_DAY"]}}


def main():
    RESULTS.mkdir(exist_ok=True)
    raw = download()
    df = collect_images(raw)
    print(f"Found {len(df):,} images in classes {sorted(df['label'].unique())}")
    df = make_split(df)
    stats = describe(df)
    weights = finetune()
    results, preds = evaluate(df, weights)
    results["dataset"] = stats
    results["training"] = {"base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ,
                           "batch": BATCH, "seed": SEED, "device": pick_device(),
                           "dataset": f"{WORKSPACE}/{PROJECT}/{VERSION}"}
    results["costs"] = cost_analysis(preds)

    print("\nTest results:")
    for name, m in results["models"].items():
        b = m["binary_at_argmax"]
        print(f"  {name:45s} acc {m['accuracy']:.3f}  sens {b['sensitivity']:.3f}  "
              f"spec {b['specificity']:.3f}")
    with open(RESULTS / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nSaved {WEIGHTS / 'best.pt'} and results/ - ready for the app.")


if __name__ == "__main__":
    main()
