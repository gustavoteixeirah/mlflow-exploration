# Iris Setosa and Versicolor Classification Model

## Overview

This document provides documentation for the "Iris Setosa and Versicolor Classification" model. The model is built to classify Iris flowers into two classes, Setosa and Versicolor, using the Support Vector Machine (SVM) algorithm. It is trained on the Iris dataset, containing petal length and petal width features.

## Dataset

The model is trained on the famous Iris dataset from scikit-learn. The dataset contains samples of three different Iris flower species: Setosa, Versicolor, and Virginica. For this model, we focus on classifying Iris flowers into Setosa and Versicolor classes. The dataset consists of petal length and petal width features for each sample.

## Model Details

The "Iris Setosa and Versicolor Classification" model is developed using the `SVC` (Support Vector Classification) algorithm from scikit-learn with a linear kernel. The SVM is a powerful classification technique that seeks to find the best decision boundary that separates the classes.

## Model Training

The SVM classifier is trained using the petal length and petal width features of the Iris flowers. We perform a binary classification by considering only two classes, Setosa and Versicolor (ignoring Virginica). The `C` hyperparameter is set to 10000.0 to enforce hard-margin classification, allowing minimal margin violations in the training data.

## Model Evaluation

To evaluate the model's performance, we use standard metrics like accuracy, precision, recall, and F1-score. Since the dataset contains only two classes, confusion matrix metrics are applicable.

## Model Autologging

MLflow's autologging feature is enabled for this model using `mlflow.sklearn.autolog()`. This ensures that all relevant model parameters, metrics, and artifacts are automatically logged and tracked in the designated MLflow tracking server.

## Conclusion

The "Iris Setosa and Versicolor Classification" model demonstrates its ability to accurately classify Iris flowers into Setosa and Versicolor classes based on petal length and petal width features. By leveraging the power of Support Vector Machines, the model achieves a robust decision boundary to differentiate between the two classes. The model can be used for further insights or integrated into larger applications for Iris flower classification tasks.
