import mlflow
import os
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["DATABRICKS_HOST"] = ""
os.environ["DATABRICKS_TOKEN"] = ""
mlflow.set_registry_uri("databricks")
mlflow.sklearn.autolog(registered_model_name="Forecasting Wind Power")


cal_housing = fetch_california_housing()
 
# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.2)
from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mlflow.set_experiment("/Users/gteicom@hotmail.com/Forecasting Wind Power")
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run():
  
  # Set the model parameters. 
  n_estimators = 100
  max_depth = 6
  max_features = 3
  
  # Create and train model.
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  
  # Use the model to make predictions on the test dataset.
  predictions = rf.predict(X_test) 

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
 
search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 2, 10, 1)),
  'n_estimators': scope.int(hp.quniform('n_estimators', 200, 1000, 100)),
  'max_features': scope.int(hp.quniform('max_features', 3, 8, 1)),
}
 
def train_model(params):
   
  # Create and train model.
  rf = RandomForestRegressor(random_state=0, **params)
  rf.fit(X_train, y_train)
  
  predictions = rf.predict(X_test)
  
  # Evaluate the model
  mse = mean_squared_error(y_test, predictions)
  
  return {"loss": mse, "status": STATUS_OK}
  
  
# spark_trials = SparkTrials()
trials = Trials()
with mlflow.start_run() as run:
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=2,
    trials=trials)


feature_importances = pd.DataFrame(rf.feature_importances_, index=cal_housing.feature_names, columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

import hyperopt
 
print(hyperopt.space_eval(search_space, best_params))
max_depth = int(hyperopt.space_eval(search_space, best_params)["max_depth"])
max_features = int(hyperopt.space_eval(search_space, best_params)["max_features"])
n_estimators = int(hyperopt.space_eval(search_space, best_params)["n_estimators"]) 
X_all_train = scaler.fit_transform(cal_housing.data)
y_all_train = cal_housing.target


with mlflow.start_run() as run:
  
  rf_new = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf_new.fit(X_all_train, y_all_train)
  
  # Save the run information to register the model later
  rf_uri = run.info.artifact_uri
  
  # Plot predicted vs known values for a quick visual check of the model and log the plot as an artifact
  rf_pred = rf_new.predict(X_all_train)
  plt.plot(y_all_train, rf_pred, "o", markersize=2)
  plt.xlabel("observed value")
  plt.ylabel("predicted value")
  plt.savefig("rfplot.png")
  mlflow.log_artifact("rfplot.png") 


import time
 
model_name = "Forecasting Wind Power"
model_uri = rf_uri+"/model"
new_model_version = mlflow.register_model(model_uri, model_name)
 
# Registering the model takes a few seconds, so add a delay before continuing with the next cell
time.sleep(5)


new_data = [[ 2.2 , -0.9,  1.05, -0.08, -0.34, 0.01,  0.74, -1.1],
            [ -0.9 , 2.6,  -1.4, -0.54, -0.86, 0.77,  0.35, -.08] ]
 
rf_model = mlflow.sklearn.load_model(f"models:/{model_name}/{new_model_version.version}")
preds = rf_model.predict(new_data)
