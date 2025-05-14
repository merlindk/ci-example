from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

app = FastAPI()
model = load("model.joblib")

class Features(BaseModel):
    data: list[list[float]]

@app.post("/predict")
def predict(features: Features):
    input_data = np.array(features.data)
    prediction = model.predict(input_data).tolist()
    return {"prediction": prediction}