
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import os


os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["DATABRICKS_HOST"] = ""
os.environ["DATABRICKS_TOKEN"] = ""
mlflow.set_registry_uri("databricks")

mlflow_client = MlflowClient()


def print_registered_models_info(r_models):
    if (r_models):
        print("Deleting currently registered models:")
        for rm in r_models:
            print("deleting name: {}".format(rm.name))
            mlflow_client.delete_registered_model(rm.name)

print_registered_models_info(mlflow_client.search_registered_models())


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def createNewModel(file_path, model_name, model_tags):
    model_description = read_file(file_path)
    
    mlflow_client.create_registered_model(model_name, model_tags, model_description)
    print("Created: ", model_name)


description_file_path = 'descriptions/Diabetes.md'
model_name = "Diabetes"
model_tags = {
    "framework": "scikit-learn",
    "version": "1.0",
    "author": "John Doe",
    "created_at": "2023-08-04"
}

createNewModel(description_file_path, model_name, model_tags)

description_file_path = 'descriptions/Forecasting Wind Power.md'
model_name = "Forecasting Wind Power"
model_tags = {
    "framework": "scikit-learn",
    "author": "John Doe",
    "created_at": "2023-08-04",
    "algorithm": "RandomForestRegressor",
    "problem_type": "Regression",
    "n_estimators": 100,
    "max_depth": 6,
    "max_features": 3
}

createNewModel(description_file_path, model_name, model_tags)


description_file_path = 'descriptions/Iris.md'
model_name = "Iris"
model_tags = {
    "framework": "scikit-learn",
    "author": "Jane Smith",
    "created_at": "2023-09-15",
    "algorithm": "SupportVectorMachine",
    "problem_type": "BinaryClassification",
    "C": 10000.0,
    "kernel": "linear"
}

createNewModel(description_file_path, model_name, model_tags)