import mlflow

run_id = "4eef75403cd646989ab68f5072d60b0d"

model_uri = f"runs:/{run_id}/model"

mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri,
    dst_path="model"
)

print("Model exported to ./model")
