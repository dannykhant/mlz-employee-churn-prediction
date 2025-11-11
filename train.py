import pandas as pd

import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer

from xgboost import XGBClassifier

output_file = "model_v20251111.bin"
numerical = ["age", "experience", "job_tenure"]
categorical = ["education", "city", "payment_tier", "gender", "ever_benched"]


def preprocessng(df):
    cat_cols = df.dtypes[df.dtypes == "object"].index
    for c in cat_cols:
        df[c] = df[c].str.lower().str.replace(" ", "_")

    df.columns = df.columns.str.lower()
    cols = {"joiningyear": "joining_year", 
            "paymenttier": "payment_tier",
            "everbenched": "ever_benched",
            "experienceincurrentdomain": "experience",
            "leaveornot": "churn"
    }
    df = df.rename(columns=cols)

    payment_class = {
        1: "high",
        2: "medium",
        3: "low"
    }
    df.payment_tier = df.payment_tier.map(payment_class)

    df["job_tenure"] = 2018 - df["joining_year"]

    return df


def validation_framework(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    return df_full_train, df_test


def train_xgb(df_train, y_train, n=10, eta=0.3, depth=6, leaf=1):
    dv = DictVectorizer(sparse=True)
    dict_train = df_train[numerical + categorical].to_dict(orient="records")
    X_train = dv.fit_transform(dict_train)

    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=n,
        learning_rate=eta,
        max_depth=depth,
        min_child_weight=leaf
    )
    model.fit(X_train, y_train)

    return dv, model


def predict_xgb(df_val, dv, model):
    dict_val = df_val[numerical + categorical].to_dict(orient="records")
    X_val = dv.transform(dict_val)
    
    y_pred = model.predict_proba(X_val)[:, 1]

    return y_pred


def get_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred > 0.5)
    precision = precision_score(y_true, y_pred > 0.5)
    recall = recall_score(y_true, y_pred > 0.5)
    auc = roc_auc_score(y_true, y_pred)

    return (accuracy, precision, recall, auc)


def cross_validation(df_full_train):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    print(("accuracy", "precision", "recall", "auc"))

    for train_idx, val_idx in kf.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train_xgb(df_train, y_train)
        y_pred = predict_xgb(df_val, dv, model)

        score = get_scores(y_val, y_pred)
        print(score)


def final_model(df_full_train, df_test):
    y_full_train = df_full_train.churn.values
    y_test = df_test.churn.values

    dv, model = train_xgb(df_full_train, y_full_train)
    y_pred = predict_xgb(df_test, dv, model)

    score = get_scores(y_test, y_pred)
    print(("accuracy", "precision", "recall", "auc"))
    print(f"Final score: {score}")

    return dv, model


def export_model(output_file, dv, model):
    with open(output_file, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    
    print(f"Model {output_file} has been saved.")


def main():
    print("Employee Churn Prediction: Model Training")

    print("### Loading the data...")
    df = pd.read_csv("employee-data.csv")

    print("### Preprocsssing the data...")
    df = preprocessng(df)

    print("### Validation framework...")
    df_full_train, df_test = validation_framework(df)

    print("### Cross validation...")
    cross_validation(df_full_train)

    print("### Final model...")
    dv, model = final_model(df_full_train, df_test)

    print("### Saving model...")
    export_model(output_file, dv, model)


if __name__ == "__main__":
    main()
