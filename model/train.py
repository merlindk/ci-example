from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=201)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {acc:.2f}")