# Meridian Claim Anomaly Detection POC

> A Forward Deployed Engineering (FDE) proof of concept. This repository contains both the **pre-transformation** (before) and **post-transformation** (after) states of the solution, so the impact of the work can be seen side by side.

---

## 1. Overview

| Item | Details |
|------|---------|
| **POC Name** | Meridian Claim Anomaly Detection POC |
| **Scenario / Customer** | Meridian General Insurance (illustrative scenario): ~4,000 UK motor claims a month |
| **Owner** | Adi Raj |
| **Status** | Completed |
| **Program** | FDE Academy, Sprint 2 (Days 26–28): Anomaly Detection & Risk Scoring |

**One-line summary:** Rules plus a gradient-boosting model can rank motor claims for human review, using honest evaluation (PR-AUC, reviews per day) and routing that never states a decision.

---

## 2. Problem Statement

Over six months, **38 valid-looking claims were paid into 4 bank accounts**. No individual claim looked wrong; the pattern only appears when many claims are viewed together.

- Single-claim checks cannot see cross-case patterns (shared bank accounts, devices, garages).
- Fraud is rare (~1.8%) and comes in several shapes, so accuracy and ROC AUC alone give a flattering, misleading picture.
- Automated approve/reject based on a risk score would be automated decision-making with a significant effect on a person, which carries legal obligations.

---

## 3. Pre-Transformation (Before)

The starting state is the learner handout and an un-run notebook: the exercises, prompts and acceptance criteria, but no results, no trained model and no scoring package ready to register.

| Aspect | Current State |
|--------|---------------|
| Process | Guided exercises (event/entities/variables, data generation, rules, training, thresholds, routing, registration); nothing executed |
| Model / Tooling | Colab notebook with empty outputs; scoring scripts only (`score.py`, `register_as_mlflow.py`, `conda.yaml`) |
| Cost | No Azure spend; no baseline measured |
| Latency / Performance | Not measured |
| Accuracy / Quality | No metrics; syllabus target of ROC AUC > 0.85 is an untested assumption |
| Pain points | Syllabus assumes 7.5% fraud and mixes Azure ML with fraud-product terms (detector, event type); no model card; no evidence for the staffing conversation |

**Location in repo:** [Pre-Transformation/](Pre-Transformation/)

---

## 4. Post-Transformation (After)

The notebook was run end to end on realistic synthetic data. It produces a trained model, a model card, a threshold-to-staffing table, tested routing and a deployable scoring package.

| Aspect | New State |
|--------|-----------|
| Process | 8-variable claim event scored by additive rules (with the human sentence behind each) plus a model; routed to a review queue |
| Model / Tooling | scikit-learn `GradientBoostingClassifier`, saved as `meridian_model/model.pkl`; Azure ML scoring package in `score/` |
| Cost | Exercises 1–8 run locally at zero Azure cost; registration step is optional (skipped when no workspace is connected) |
| Latency / Performance | Not benchmarked (batch scoring on 4,000 cases runs in the notebook session) |
| Accuracy / Quality | ROC AUC 0.761, PR-AUC 0.357 (guessing baseline 0.018, ~19.5x lift) on 22 test fraud cases |
| Benefits | Honest metrics, reviews-per-day numbers for the operations manager, routes that cannot decide outcomes, fail-towards-human fallback, model card with limitations |

**Location in repo:** [Post-Transformation/](Post-Transformation/)

---

## 5. Before vs. After

| Metric | Pre-Transformation | Post-Transformation | Change |
|--------|--------------------|---------------------|--------|
| Fraud rate used | 7.5% (syllabus assumption) | 1.8% (72 of 4,000 cases) | Realistic imbalance |
| ROC AUC | Not measured | 0.761 (target was > 0.85, not reached) | Measured, not tuned |
| PR-AUC | Not measured | 0.357 vs 0.018 baseline | ~19.5x over guessing |
| Fraud caught, rules only | Not measured | 4 of 22 (18.2% recall, 66.7% precision) | Baseline established |
| Fraud caught, model only | Not measured | 9 of 22 (40.9% recall, 34.6% precision) | +22.7 pts recall vs rules |
| Fraud caught, rules OR model | Not measured | 10 of 22 (45.5% recall, 34.5% precision) | Best recall, ~3.2 reviews/day |
| Model card | None | Purpose, "not for", limitations, metrics | Governance artefact added |

Threshold trade-off for the operations manager (model only, ~133 claims/day):

| Threshold | Precision | Recall | Flagged/day | Real/day | Wasted/day |
|-----------|-----------|--------|-------------|----------|------------|
| 0.03 | 24.4% | 45.5% | 4.6 | 1.1 | 3.4 |
| 0.10 | 34.6% | 40.9% | 2.9 | 1.0 | 1.9 |
| 0.20 | 40.9% | 40.9% | 2.4 | 1.0 | 1.4 |
| 0.50 | 75.0% | 27.3% | 0.9 | 0.7 | 0.2 |

Source: executed notebook [Sprint2_Anomaly_Detection_Colab.ipynb](Post-Transformation/Sprint2_Anomaly_Detection_Colab.ipynb).

---

## 6. Approach

1. **Discover** – define the event (a claim arriving), entities (claimant, policy, garage, bank account) and 8 variables, two of them relational; check availability at scoring time and proxy risk.
2. **Baseline** – generate 4,000 claims at a 1.8% fraud rate with three overlapping fraud patterns plus unlearnable cases; measure rules-only performance.
3. **Evaluate** – stratified split before exploration; report ROC AUC, PR-AUC and the guessing baseline; compare rules only, model only, OR and AND.
4. **Build** – gradient-boosting classifier, additive rule engine that returns fired rules, routing (`straight_through` / `standard_review` / `priority_review`) with unit tests, and an Azure ML scoring package.
5. **Measure** – turn thresholds into flagged, real and wasted reviews per day.
6. **Hand off** – model card with "not for" statement and limitations; workbook and PDF for the case study.

### Decision Criteria

| Criterion | Weight | Why |
|---|---|---|
| Honest evaluation | High | With 1.8% positives, ROC AUC flatters; PR-AUC and test-set fraud count must be shown |
| Governance | High | The system supports a decision and never states one; no approve/reject routes |
| Reviewer workload | High | The threshold is a staffing decision for the client, not a metric optimum |
| Explainability | Medium | Handlers need the rules that fired and plain-language reasons, not a bare score |

---

## 7. Repository Structure

```
Assignment17/
├── README.md
├── Pre-Transformation/                    # "Before" state
│   ├── Sprint2_Anomaly_Detection_Colab.ipynb      # un-run notebook
│   ├── Sprint2_Anomaly_Detection_Generate_and_Govern.md
│   ├── Precision_Recall_Threshold_Demo.ipynb
│   ├── FDE_Anomaly_Detection_Case_Study_Workbook.docx
│   ├── Sprint2_Anomaly_Detection_FDE_Mindset.pdf
│   └── score/                             # score.py, register_as_mlflow.py, conda.yaml
└── Post-Transformation/                   # "After" state
    ├── Sprint2_Anomaly_Detection_Colab.ipynb      # executed, with results
    ├── Sprint2_Anomaly_Detection_Generate_and_Govern.md
    ├── Precision_Recall_Threshold_Demo.ipynb
    ├── FDE_Anomaly_Detection_Case_Study_Workbook.docx
    ├── Sprint2_Anomaly_Detection_FDE_Mindset.pdf
    ├── meridian_model/                    # model.pkl, model_card.json
    └── score/                             # score.py, register_as_mlflow.py, conda.yaml
```

---

## 8. Getting Started

### Prerequisites

- Python 3.10+ (or Google Colab)
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- Optional, for model registration only: an Azure ML workspace plus `azure-ai-ml` and `azure-identity` (sign-in is by identity, device code in Colab; there is no API key)

### Setup

```powershell
cd Post-Transformation
py -3 -m venv .venv
.venv\Scripts\pip install scikit-learn pandas numpy matplotlib jupyter
```

### Run

Open `Sprint2_Anomaly_Detection_Colab.ipynb` in Jupyter or Colab and run all cells. Exercise 9 (Azure registration) is skipped automatically when no workspace is connected.

To score locally with the saved model, call `init()` and `run()` from `score/score.py` with `AZUREML_MODEL_DIR` pointing at `meridian_model`'s parent folder, passing `{"cases": [{...8 features...}]}`.

---

## 9. Results & Learnings

- **Key result:** ROC AUC 0.761 and PR-AUC 0.357 against a 0.018 guessing baseline; the syllabus target of ROC AUC > 0.85 was not reached and was deliberately not tuned for.
- **What worked:** realistic, overlapping synthetic data; rules and model complement each other (rules only catch the already-noticed "ring" pattern); fallback routes unscored cases to a human, and all routing tests pass.
- **What didn't:** rules alone caught 0% of the opportunist, staged and invisible patterns; the model still misses over half of fraud at workable thresholds.
- **Recommendations:** take the threshold table to the operations manager and let them choose based on reviewer capacity; re-measure on real claims before production; review `garage_claim_share_90d` per geography for proxy risk.

> Caveats: data is synthetic, only 22 fraud cases are in the test set (metrics are noisy, do not trust the third decimal), and selection bias means fraud that was never detected is invisible to the model.

---

## 10. Next Steps

- [ ] Name the operations manager who agrees the thresholds (`chosen_by` in the model card is still TODO)
- [ ] Fill in `reviewed_by` in the model card
- [ ] Register the model in an Azure ML workspace and deploy `score/`
- [ ] Re-evaluate on real claims data and review proxy risk per geography

---

## 11. Contact

Adi Raj · adiraj2503@gmail.com
