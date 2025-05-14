import time
from joblib import load
from model import train

def test_model_latency(model_path="model.joblib", threshold=1.0):
    print(f"Testing latency of model at: {model_path}")
    X_train, X_test, y_train, y_test = train.load_data()
    model = load(model_path)
    start = time.time()
    model.predict(X_test)
    elapsed = time.time() - start
    print(f"Model inference time: {elapsed:.4f} seconds")
    if elapsed > threshold:
        raise Exception(f"Latency too high: {elapsed:.4f}s (limit: {threshold}s)")

if __name__ == "__main__":
    test_model_latency()