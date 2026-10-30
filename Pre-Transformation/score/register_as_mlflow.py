"""
The easier route: register the model as MLflow instead of a custom model.

Azure then generates the scoring script and the environment for you, and the
"Code + environment" step you are stuck on disappears entirely.

Run this in the same Colab notebook, after the model is trained.
"""
import mlflow, mlflow.sklearn, json
from pathlib import Path

# 1. Save in MLflow format. This writes the model PLUS a conda spec describing
#    the exact library versions, which is what removes the environment problem.
MLFLOW_DIR = "meridian_mlflow_model"
mlflow.sklearn.save_model(
    sk_model=model,                       # from the training cell
    path=MLFLOW_DIR,
    input_example=cases[FEATURES].head(2),   # gives Azure the input schema
)
print(f"saved to {MLFLOW_DIR}/")
print(sorted(p.name for p in Path(MLFLOW_DIR).iterdir()))

# 2. Register it as an MLFLOW_MODEL, not a CUSTOM_MODEL.
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

registered = ml_client.models.create_or_update(Model(
    path=MLFLOW_DIR,
    type=AssetTypes.MLFLOW_MODEL,          # <-- the line that matters
    name="meridian-claim-anomaly-mlflow",
    description="Ranks motor claims for HUMAN review. Never decides.",
    tags={"decides_outcomes": "no", "sprint": "2"},
))
print(f"registered {registered.name} v{registered.version} as MLFLOW_MODEL")
print("Deploy this one - the portal will not ask for a scoring script.")
