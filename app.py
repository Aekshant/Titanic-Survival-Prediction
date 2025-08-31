from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Load model
model = joblib.load("titanic_model.pkl")

# Define input schema
class Passenger(BaseModel):
    Pclass: int
    Age: float
    Fare: float
    Sex: int      # 0 = female, 1 = male
    Embarked: int # 0 = C, 1 = Q, 2 = S

app = FastAPI(title="Titanic Survival Prediction API")
Instrumentator().instrument(app).expose(app)

@app.post("/predict")
def predict(passenger: Passenger):
    data = np.array([[passenger.Pclass, passenger.Age, passenger.Fare, passenger.Sex, passenger.Embarked]])
    
    prediction = model.predict(data)[0]
    return {"survived": int(prediction)}
