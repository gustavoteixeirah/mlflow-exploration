import os
import mlflow
from mlflow import MlflowClient


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

mlflow.set_registry_uri("databricks")
client = MlflowClient()

def transition_versions_to_archived(model_name, version_numbers):
    mlflow.sklearn.autolog(registered_model_name=model_name)
    for version in version_numbers:
        try:
            client.transition_model_version_stage(
                name=model_name, version=version, stage="Archived"
            )
        except Exception as e:
            print(f"Error occurred during transition of version {version}: {e}")
            continue

if __name__ == "__main__":
    model_name = "Iris"
    versions_to_archive = [1, 2, 3]
    transition_versions_to_archived(model_name, versions_to_archive)
