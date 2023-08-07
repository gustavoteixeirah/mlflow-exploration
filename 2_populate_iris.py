import os
import mlflow
from mlflow import MlflowClient
import os

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "databricks")
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")
mlflow.set_registry_uri("databricks")
mlflow.sklearn.autolog(registered_model_name="Iris")

client = MlflowClient()

from sklearn import datasets

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

from sklearn.svm import SVC

mlflow.set_experiment("/Users/gteicom@hotmail.com/Iris")

with mlflow.start_run():
    svm_clf = SVC(kernel="linear", C=10000.0)
    svm_clf.fit(X, y)


client.transition_model_version_stage(
name="Iris", version=1, stage="Production"
)

with mlflow.start_run():
    svm_clf = SVC(kernel="sigmoid", C=20000.0)
    svm_clf.fit(X, y)

client.transition_model_version_stage(
    name="Iris", version=2, stage="Archived"
)

with mlflow.start_run():
    svm_clf = SVC(kernel="rbf", C=5000.0)
    svm_clf.fit(X, y)

client.transition_model_version_stage(
    name="Iris", version=3, stage="Staging"
)

with mlflow.start_run():
    svm_clf = SVC(kernel="poly", C=8080.0)
    svm_clf.fit(X, y)

