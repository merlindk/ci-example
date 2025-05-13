import numpy as np
from src.model import normalize_data, train_model

def test_normalize_data():
    data = [10, 20, 30]
    norm = normalize_data(data)
    assert round(np.mean(norm), 5) == 0
    assert round(np.std(norm), 5) == 1

def test_train_model():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model, rmse = train_model(X, y)
    assert rmse < 1.0