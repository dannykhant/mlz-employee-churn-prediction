import pickle

model_file = "model_v20251111.bin"
json_data = {
    "education": "bachelors", 
    "joining_year": 2016, 
    "city": "pune", 
    "payment_tier": "high", 
    "age": 26, 
    "gender": "female", 
    "ever_benched": "no", 
    "experience": 4, 
    "job_tenure": 2
}


def load_model(model_file):
    with open(model_file, "rb") as f_in:
        dv, model = pickle.load(f_in)

    return dv, model


def predict(json_data, dv, model):
    X = dv.transform([json_data])
    
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


def main():
    dv, model = load_model(model_file)
    y_pred = predict(json_data, dv, model)
    churn = y_pred[0] >= 0.5
    print(f"Churn: {churn}, Churn_proba: {y_pred[0]}")


if __name__ == "__main__":
    main()
