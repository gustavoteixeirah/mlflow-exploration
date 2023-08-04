# Diabetes Linear Regression Model Documentation

## Overview

This document provides an overview of the linear regression model developed for predicting diabetes progression using the diabetes dataset. The model is built using scikit-learn and is auto-logged using MLflow, a platform for managing the machine learning lifecycle.

## Dataset

The model is trained on the [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) from scikit-learn. This dataset consists of ten baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements for 442 diabetes patients. The target variable is a quantitative measure of disease progression one year after baseline.

## Model Details

### Model Type

The diabetes linear regression model is a supervised machine learning model that uses linear regression to establish a relationship between a single feature (blood serum measurement) and the target variable (diabetes progression).

### Model Training

The model is trained using scikit-learn's `LinearRegression` class. The training data is split into training and testing sets, where 20% of the data is used for testing. The selected feature for training is the third blood serum measurement (diabetes_X[:, np.newaxis, 2]).

### Model Evaluation

The model's performance is evaluated using the following metrics:
- Mean Squared Error (MSE): The mean squared difference between the actual and predicted values of diabetes progression on the test set.
- Coefficient of Determination (R-squared): Also known as the coefficient of determination, this metric measures the proportion of the variance in the target variable that is predictable from the feature used by the model.

### Model Autologging

MLflow's auto-logging feature is enabled for this model using `mlflow.sklearn.autolog()`. This allows the model's hyperparameters, metrics, and artifacts to be automatically logged and stored in the designated MLflow tracking server.

## Results

After training the model and evaluating its performance on the test set, the following results were obtained:

- Coefficients: The coefficients of the linear regression model represent the change in the predicted diabetes progression for a one-unit change in the selected feature (third blood serum measurement).
- Mean Squared Error: The model's mean squared error on the test set is presented as a quantitative measure of its predictive accuracy.
- Coefficient of Determination: The R-squared value indicates the proportion of the variance in diabetes progression that is explained by the linear relationship with the selected feature.

## Model Usage

Once the model is trained and logged, it can be easily deployed and integrated into other applications or services for diabetes progression prediction.

### Model Dependencies

To use this model, you'll need the following dependencies:
- Python (>= 3.6)
- scikit-learn (>= 0.24.2)
- mlflow (>= 1.18.0)
- numpy (>= 1.19.5)

### Loading the Model

The trained model can be loaded using the MLflow API and used to make predictions.

```python
import mlflow.sklearn

# Load the trained model
model = mlflow.sklearn.load_model("Diabetes")

# Use the model to make predictions
predictions = model.predict(diabetes_X_test)
```