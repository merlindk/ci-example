from model import train
from joblib import dump, load
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import time
import os

def test_model_accuracy():
    X_train, X_test, y_train, y_test = train.load_data()
    model = train.train_model(X_train, y_train)
    accuracy = train.evaluate_model(model, X_test, y_test)
    assert accuracy > 0.7, "Model accuracy should be greater than 70%"

def test_classification_report():
    X_train, X_test, y_train, y_test = train.load_data()
    model = train.train_model(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test))
    print("Classification Report:\n", report)

def test_model_reproducibility():
    X_train, X_test, y_train, y_test = train.load_data()
    model = train.train_model(X_train, y_train)
    dump(model, "temp_model.joblib")
    loaded_model = load("temp_model.joblib")
    assert (model.predict(X_test) == loaded_model.predict(X_test)).all(), "Model predictions differ after loading"
    os.remove("temp_model.joblib")

def test_model_with_noise():
    X_train, X_test, y_train, y_test = train.load_data()
    model = train.train_model(X_train, y_train)
    X_test_noisy = X_test + np.random.normal(0, 0.1, X_test.shape)
    noisy_accuracy = accuracy_score(y_test, model.predict(X_test_noisy))
    assert noisy_accuracy > 0.5, f"Model not robust to noise: {noisy_accuracy:.2f}"

def test_input_output_contract():
    X_train, X_test, y_train, y_test = train.load_data()
    model = train.train_model(X_train, y_train)
    sample = X_test[0]
    assert isinstance(sample, np.ndarray)
    pred = model.predict(sample.reshape(1, -1))
    assert pred.shape == (1,), "Unexpected prediction shape"