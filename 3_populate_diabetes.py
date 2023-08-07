import mlflow
from mlflow import MlflowClient
import os

from sklearn.ensemble import RandomForestRegressor
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "databricks")
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")
mlflow.set_registry_uri("databricks")
mlflow.sklearn.autolog(registered_model_name="Diabetes")
mlflow.set_experiment("/Users/gteicom@hotmail.com/Diabetes")
tags1 = {
    "engineering": "ML Platform",
    "release.candidate": "RC1",
    "release.version": "2.2.0",
}
tags2 = {
    "engineering": "DS Platform 2",
    "release.candidate": "RC9",
    "release.version": "14.5.2",
}

tags3 = tags1.update(tags2)

import mlflow.sklearn
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

diabetes_X = diabetes_X[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]

diabetes_y_test = diabetes_y[-20:]

with mlflow.start_run() as run:
    mlflow.set_experiment_tags(tags1)
    lr = linear_model.LinearRegression()
    lr.fit(diabetes_X_train, diabetes_y_train)


diabetes_y_pred = lr.predict(diabetes_X_test)

import matplotlib.pyplot as plt
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())
plt.savefig("lr_plot.png")
mlflow.log_artifact("lr_plot.png") 
mlflow.end_run()


n_estimators = 100
max_depth = 6
max_features = 3

with mlflow.start_run():
    mlflow.set_experiment_tags(tags2)
    rfr = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rfr.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = rfr.predict(diabetes_X_test)

import matplotlib.pyplot as plt
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())
plt.savefig("rfr_plot.png")
mlflow.log_artifact("rfr_plot.png") 
mlflow.end_run()

n_estimators = 500
max_depth = 4
max_features = 2

with mlflow.start_run():
    mlflow.log_param("my", "param")
    mlflow.log_metric("score", 100)
    mlflow.set_experiment_tags(tags3)
    rfr = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rfr.fit(diabetes_X_train, diabetes_y_train)
    diabetes_y_pred = rfr.predict(diabetes_X_test)

import matplotlib.pyplot as plt
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())
plt.savefig("rfr_plot2.png")
mlflow.log_artifact("rfr_plot2.png") 
mlflow.end_run()