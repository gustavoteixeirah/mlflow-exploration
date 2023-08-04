# Forecasting Wind Power Model Documentation

## Overview

This document provides documentation for the "Forecasting Wind Power" model. The model is built to predict wind power using the RandomForestRegressor algorithm and is trained on the California housing dataset.

## Model Details

The "Forecasting Wind Power" model is developed using the `RandomForestRegressor` algorithm from scikit-learn. It aims to predict wind power based on various input features from the California housing dataset.

## Dataset

The model is trained on the California housing dataset obtained from scikit-learn's `fetch_california_housing` function. The dataset contains various features, including average occupancy, average income, and geographical coordinates, used to predict the target variable, wind power.

## Model Training

The model is trained using the RandomForestRegressor with hyperparameters:
- `n_estimators`: The number of trees in the forest.
- `max_depth`: The maximum depth of the tree.
- `max_features`: The number of features to consider when looking for the best split.

The training dataset is split into 80% training and 20% testing sets. The input features are scaled using `StandardScaler` before training.

## Hyperparameter Tuning

Hyperparameter tuning is performed using the Hyperopt library. A search space is defined for `max_depth`, `n_estimators`, and `max_features`. The `fmin` function from Hyperopt's `tpe` algorithm is used to search for the optimal hyperparameters based on the mean squared error (MSE) loss.

## Model Evaluation

The model's performance is evaluated based on the mean squared error (MSE) between the actual wind power values and the predicted values on the test dataset.

## Model Versioning

The best model with the tuned hyperparameters is registered and versioned using MLflow. The registered model is given the name "Forecasting Wind Power," and the best model version is saved.

## Predictions

The model can be used to make predictions on new data. To do so, load the best model version using the `mlflow.sklearn.load_model` function, and then provide new data to obtain the predicted wind power values.

## Dependencies

The following libraries and versions are required to run this model:
- mlflow (>= 1.20.2)
- scikit-learn (>= 0.24.2)
- pandas (>= 1.3.1)
- matplotlib (>= 3.4.2)
- hyperopt (>= 0.2.5)

Please ensure you have these dependencies installed before running the model.


The model's performance is evaluated based on the mean squared error (MSE) between the actual wind power values and the predicted values on the test dataset.

## Hyperparameter Tuning

Hyperparameter tuning is performed using the Hyperopt library to find the best combination of hyperparameters for the RandomForestRegressor.

## Conclusion

The "Forecasting Wind Power" model is a RandomForestRegressor-based model trained on the California housing dataset. By tuning the hyperparameters using Hyperopt, the model's performance is optimized to predict wind power accurately. The model can be easily versioned, managed, and used for making predictions on new data using MLflow.
