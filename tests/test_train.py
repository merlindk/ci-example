from model import train
import numpy as np

def test_model_accuracy():
    X_train, X_test, y_train, y_test = train.load_data()
    model = train.train_model(X_train, y_train)
    accuracy = train.evaluate_model(model, X_test, y_test)
    assert accuracy > 0.7, "Model accuracy should be greater than 70%"