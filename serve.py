from fastapi import FastAPI
from pydantic import BaseModel

import pickle



def load_model(model_file):
    with open(model_file, "rb") as f_in:
        dv, model = pickle.load(f_in)

    return dv, model


class Employee(BaseModel):
    education: str = "bachelors"
    joining_year: int = 2016
    city: str = "pune"
    payment_tier: str = "high"
    age: int = 26
    gender: str = "female"
    ever_benched: str = "no"
    experience: int = 4
    job_tenure: int = 2


class Churn(BaseModel):
    churn: bool
    churn_proba: float


app = FastAPI()
dv, model = load_model("model_v20251111.bin")


@app.get("/health")
async def health() -> dict:
    return {"status": "running"}


@app.post("/predict")
async def predict(employee: Employee) -> Churn:
    X_employee = dv.transform(employee.model_dump())
    y_pred = model.predict_proba(X_employee)[0, 1]
    churn = y_pred >= 0.5

    return {
        "churn": bool(churn),
        "churn_proba": round(float(y_pred), 3)
    }
