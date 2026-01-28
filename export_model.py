import mlflow

run_id = "608fb417d32a4e728d4d4f284b14ebf6"

model_uri = f"runs:/{run_id}/model"

mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri,
    dst_path="model"
)

print("Model exported to ./model")
