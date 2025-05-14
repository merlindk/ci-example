# 🧠 ML CI/CD Example with FastAPI Deployment

This repository demonstrates a complete **machine learning pipeline** with **CI/CD**, including:

- ✅ Model training using scikit-learn
- ✅ Unit and robustness tests (latency, reproducibility, accuracy)
- ✅ Continuous Integration (CI) with GitHub Actions
- ✅ Continuous Delivery (CD) via automatic container deployment
- ✅ REST API for inference (FastAPI + Docker)
- ✅ Hosted on [Render](https://render.com)
- ✅ Remote API testing script

---

## 🚀 Features

### 🔁 Continuous Integration

- Runs on pull requests via GitHub Actions
- Includes:
  - Accuracy tests
  - Latency tests
  - Noise robustness
  - Input/output contract tests
- Coverage summary posted as a PR comment

### 🚢 Continuous Delivery

- Triggered on pushes to `main`
- Trains and saves a model (`model.joblib`)
- Runs latency test
- Commits model to repository
- Render automatically redeploys Docker container

---

## 🛠 Project Structure

```
ml-ci-example/
├── model/
│   └── train.py         # Model training script
├── tests/
│   ├── test_train.py    # Basic unit tests
├── it/
│   └── latency_check.py # Standalone latency test
├── serve.py             # FastAPI model server
├── test_api.py          # Script to test deployed API
├── Dockerfile           # Container for deployment
├── requirements.txt
└── .github/
    └── workflows/
        ├── ci.yml       # CI workflow
        └── cd.yml       # CD workflow
```

---

## 🌐 Deployed API

Your FastAPI model is live on Render:

> **POST** [`/predict`](https://your-app-name.onrender.com/predict)

### Request Example:

```json
{
  "data": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3]
  ]
}
```

### Response:

```json
{
  "prediction": [0, 2]
}
```

Visit [`/docs`](https://your-app-name.onrender.com/docs) for interactive API documentation.


---

## 📜 License

MIT – free to use, fork, and deploy.
