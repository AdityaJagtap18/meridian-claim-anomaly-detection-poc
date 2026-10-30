"""
score.py — scoring script for meridian-claim-anomaly

Azure ML calls init() once when the container starts, then run() for every
request. Both names are fixed; do not rename them.

GOVERNANCE NOTE — read before you edit this file.

This endpoint returns a SCORE and the REASONS behind it. It does not return
approve, reject, or any other outcome. That is deliberate and it is the Sprint 0
commitment: the system supports a decision, a person makes it.

If somebody later asks you to "just add a decision field so the caller doesn't
have to work it out", that is the moment to push back. An automated rejection
based on a risk score is automated decision-making with a significant effect on
a person, and it carries obligations this design deliberately avoids.
"""

import os
import json
import logging
import joblib
import numpy as np

# The order here MUST match the order the model was trained on. If these get
# out of sync the model still returns a number and it is silently wrong, which
# is the worst kind of bug in a scoring service.
FEATURES = [
    "declared_income",
    "estimate_vs_limit_pct",
    "days_since_incident",
    "application_velocity_30d",
    "device_fingerprint_reuse",
    "bank_account_reuse_count",
    "garage_claim_share_90d",
    "policy_age_days",
]

model = None


def init():
    """Runs once when the container starts."""
    global model

    # AZUREML_MODEL_DIR points at the mounted model. The registered folder name
    # is appended, so the .pkl is one level down. If you get FileNotFoundError,
    # log os.listdir on the paths below — folder nesting is the usual cause.
    model_root = os.getenv("AZUREML_MODEL_DIR", "")
    candidates = [
        os.path.join(model_root, "meridian_model", "model.pkl"),
        os.path.join(model_root, "model.pkl"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)

    if path is None:
        logging.error("model.pkl not found. Searched: %s", candidates)
        logging.error("AZUREML_MODEL_DIR contains: %s",
                      os.listdir(model_root) if model_root else "(unset)")
        raise FileNotFoundError("model.pkl not found - see log above")

    model = joblib.load(path)
    logging.info("model loaded from %s", path)


def run(raw_data):
    """
    Expects: {"cases": [{...one dict per case, keyed by FEATURES...}]}
    Returns: one result per case, with the score and the reasons.
    """
    try:
        payload = json.loads(raw_data)
        cases = payload.get("cases", [])
        if not cases:
            return {"error": "no cases supplied"}

        # Missing features are reported, not silently defaulted to zero.
        # A zero for bank_account_reuse_count looks exactly like a clean case.
        rows, missing_per_case = [], []
        for c in cases:
            missing = [f for f in FEATURES if f not in c]
            missing_per_case.append(missing)
            rows.append([float(c.get(f, 0.0)) for f in FEATURES])

        probs = model.predict_proba(np.array(rows))[:, 1]

        results = []
        for c, p, missing in zip(cases, probs, missing_per_case):
            score = round(float(p) * 100, 1)
            results.append({
                "case_id": c.get("case_id"),
                "anomaly_score": score,
                "scored": True,
                "missing_features": missing,
                "partial": bool(missing),
                "reasons": _reasons(c),
                # No decision. No outcome. No recommendation.
                "note": "score only - routing and the decision happen upstream",
            })
        return {"results": results}

    except Exception as e:
        # Fail loudly. A scoring service that returns a low score on error is
        # far more dangerous than one that returns an error.
        logging.exception("scoring failed")
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def _reasons(case):
    """Plain-language reasons, so the queue is work rather than numbers."""
    out = []
    if case.get("bank_account_reuse_count", 0) >= 3:
        out.append(f"bank account used by {case['bank_account_reuse_count']} claims")
    if case.get("estimate_vs_limit_pct", 0) > 0.90:
        out.append("estimate is within 10% of the policy limit")
    if case.get("days_since_incident", 0) > 21:
        out.append(f"reported {case['days_since_incident']} days after the incident")
    if case.get("policy_age_days", 9999) < 30:
        out.append(f"policy only {case['policy_age_days']} days old")
    if case.get("garage_claim_share_90d", 0) > 0.20:
        out.append(f"garage in {case['garage_claim_share_90d']:.0%} of recent claims")
    return out or ["no individual factor stood out"]
