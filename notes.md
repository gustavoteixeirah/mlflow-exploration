
### Experiments
Experiment: tracking of model runs
MLflow client...
MLflow module...

```python
# Import MLflow
import mlflow
# Create new experiment
mlflow.create_experiment("Unicorn Model")
# Tag new experiment
mlflow.set_experiment_tag("scikit-learn", "lr")
# Set the experiment
mlflow.set_experiment("Unicorn Model")
```


```python
# Import MLflow
import mlflow
# Set the experiment
mlflow.set_experiment("Unicorn Sklearn Experiment")
# Start a run
mlflow.start_run()
```

```python
r2_score = r2_score(y_test, y_pred)
# Log metrics
mlflow.log_metric("r2_score", r2_score)
# Log parameter
mlflow.log_param("n_jobs", n_jobs)
# Log the training code
mlflow.log_artifact("train.py")
```

### Active runs!
```python
run = mlflow.start_run()
```

data about metrics and parameters from the run: `run.data`

attributes used to retrieve metadata about an active run: `run.info`

Filters
```python
# Create a filter string for R-squared score
r_squared_filter = "metrics.r2_score > .70"

# Search runs
mlflow.search_runs(experiment_names=["Unicorn Sklearn Experiments", "Unicorn Other Experiments"], 
                   filter_string=r_squared_filter, 
                   order_by=["metrics.r2_score DESC"])
```