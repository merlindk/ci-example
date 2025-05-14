# ML CI Example

This repository demonstrates a simple machine learning pipeline with continuous integration using GitHub Actions.

## Features

- Logistic Regression on Iris dataset
- Unit testing with `pytest`
- CI workflow triggered on push and PR

## Running locally

```bash
pip install -r requirements.txt
python model/train.py
pytest