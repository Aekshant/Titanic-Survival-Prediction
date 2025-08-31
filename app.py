from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("titanic_model.pkl")

# Define input schema
class Passenger(BaseModel):
    Pclass: int
    Age: float
    # SibSp: int
    # Parch: int
    Fare: float
    Sex: int      # 0 = female, 1 = male
    Embarked: int # 0 = C, 1 = Q, 2 = S

app = FastAPI(title="Titanic Survival Prediction API")

@app.post("/predict")
def predict(passenger: Passenger):
    data = np.array([[passenger.Pclass, passenger.Age, passenger.Fare, passenger.Sex, passenger.Embarked]])
    
    prediction = model.predict(data)[0]
    return {"survived": int(prediction)}
