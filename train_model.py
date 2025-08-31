import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import X_train, X_test, y_train, y_test

mlflow.set_experiment("Titanic_Survival")

# Logistic Regression
with mlflow.start_run(run_name="Logistic Regression"):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    acc_log = accuracy_score(y_test, y_pred_log)

    # Log metrics and model
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", acc_log)
    mlflow.sklearn.log_model(log_reg, "log_reg_model")

# Random Forest
with mlflow.start_run(run_name="Random Forest"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # Log metrics and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.sklearn.log_model(rf, "rf_model")

    # Save the best locally too
    joblib.dump(rf, "titanic_model.pkl")
    print("✅ Best Model saved as titanic_model.pkl")
