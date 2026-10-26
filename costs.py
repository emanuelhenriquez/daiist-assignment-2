"""
Business assumptions and cost maths, shared by train.py and app.py.

The decision: a chest X-ray comes in. The model gives a "pneumonia score"
s = P(bacterial) + P(viral) = 1 - P(normal). Two thresholds turn it into
one of three outcomes:

    s <  LOW            -> PASS   (treated as normal, no pneumonia follow-up)
    LOW <= s < HIGH     -> REVIEW (a radiologist reads the X-ray)
    s >= HIGH           -> FAIL   (flagged as pneumonia, treatment starts)

With LOW == HIGH there is no review band: the model decides alone.

Every number below is an ASSUMPTION, justified in REPORT.md. The dashboard
lets you change them and watch the answer move.
"""

import numpy as np
import pandas as pd

# --- Costs in euros, per case -------------------------------------------------
COST_FN = 1500.0     # pneumonia sent home as normal: returns sicker, admission, complications
COST_FP = 100.0      # healthy patient flagged: unnecessary antibiotics + follow-up visit
COST_REVIEW = 10.0   # radiologist reads one X-ray (~3 min of a ~€200/h specialist)
COST_TP = 0.0        # correct decisions are the baseline we measure mistakes against
COST_TN = 0.0

# --- Operating context --------------------------------------------------------
BASE_RATE = 0.25        # share of X-rays in real use that show pneumonia
CASES_PER_DAY = 150     # chest X-rays per day at the clinic
MINUTES_PER_REVIEW = 3  # radiologist time to read one X-ray
REVIEW_HOURS_PER_DAY = 6  # hours one radiologist can spend on these reads per day

# Assumption: a radiologist's reading of a reviewed case is correct.
# (Real radiologists miss some cases too; see REPORT.md limitations.)


def rates(y_pos: np.ndarray, score: np.ndarray, low: float, high: float) -> dict:
    """Share of positives / negatives that end up PASS, REVIEW, FAIL."""
    y_pos = np.asarray(y_pos).astype(bool)
    score = np.asarray(score)
    high = max(high, low)
    passed = score < low
    failed = score >= high
    review = ~passed & ~failed
    pos, neg = y_pos, ~y_pos
    npos, nneg = max(pos.sum(), 1), max(neg.sum(), 1)
    return {
        "pos_pass": (passed & pos).sum() / npos,      # false negatives
        "pos_review": (review & pos).sum() / npos,
        "pos_fail": (failed & pos).sum() / npos,      # true positives
        "neg_pass": (passed & neg).sum() / nneg,      # true negatives
        "neg_review": (review & neg).sum() / nneg,
        "neg_fail": (failed & neg).sum() / nneg,      # false positives
    }


def expected_cost(r: dict, base_rate=BASE_RATE, c_fn=COST_FN, c_fp=COST_FP,
                  c_review=COST_REVIEW, c_tp=COST_TP, c_tn=COST_TN) -> float:
    """Expected euros per case, re-weighted to the production base rate.

    Uses rates *within* each true class, so the dataset's own class mix
    (mostly pneumonia) does not leak into the answer.
    """
    pos_cost = r["pos_pass"] * c_fn + r["pos_review"] * c_review + r["pos_fail"] * c_tp
    neg_cost = r["neg_fail"] * c_fp + r["neg_review"] * c_review + r["neg_pass"] * c_tn
    return base_rate * pos_cost + (1 - base_rate) * neg_cost


def review_share(r: dict, base_rate=BASE_RATE) -> float:
    return base_rate * r["pos_review"] + (1 - base_rate) * r["neg_review"]


def single_threshold_sweep(y_pos, score, grid=None, **cost_kw) -> pd.DataFrame:
    """Model decides alone: one threshold, no review band."""
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2) if grid is None else grid
    rows = []
    for t in grid:
        r = rates(y_pos, score, t, t)
        rows.append({"threshold": t, "cost_per_case": expected_cost(r, **cost_kw),
                     "fn_rate": r["pos_pass"], "fp_rate": r["neg_fail"]})
    return pd.DataFrame(rows)


def two_threshold_search(y_pos, score, grid=None, **cost_kw) -> pd.DataFrame:
    """Every (low, high) pair with low <= high: pass / review / fail."""
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2) if grid is None else grid
    rows = []
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            r = rates(y_pos, score, lo, hi)
            rows.append({"low": lo, "high": hi,
                         "cost_per_case": expected_cost(r, **cost_kw),
                         "review_share": review_share(r, cost_kw.get("base_rate", BASE_RATE))})
    return pd.DataFrame(rows)


def staff_needed(review_share_value, cases_per_day=CASES_PER_DAY,
                 minutes=MINUTES_PER_REVIEW, hours=REVIEW_HOURS_PER_DAY) -> dict:
    reviews = review_share_value * cases_per_day
    minutes_total = reviews * minutes
    return {"reviews_per_day": reviews, "review_hours_per_day": minutes_total / 60,
            "radiologists_needed": minutes_total / 60 / hours}
