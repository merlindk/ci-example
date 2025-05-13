import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn

def normalize_data(data):
    """Normaliza una lista de valores numéricos (media 0, varianza 1)."""
    arr = np.array(data)
    return (arr - arr.mean()) / arr.std()

def train_model(X, y):
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    """Entrena un modelo de regresión lineal y retorna el modelo y su RMSE."""
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = mean_squared_error(y, predictions, squared=False)

    with mlflow.start_run():
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "linear_model")

    return model, rmse