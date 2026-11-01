import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ---------------------------------------------------------
# 1. DATA GENERATION & PREPARATION
# ---------------------------------------------------------
def generate_and_prepare_data(filepath="student_data.csv", num_samples=500):
    np.random.seed(42)

    student_ids = np.arange(101, 101 + num_samples)
    attendance = np.clip(np.random.normal(75, 15, num_samples), 35, 100).round(1)
    internal_marks = np.clip(np.random.normal(68, 16, num_samples), 20, 100).round(1)
    assignment_scores = np.clip(np.random.normal(72, 15, num_samples), 25, 100).round(1)
    prev_sem_marks = np.clip(np.random.normal(66, 17, num_samples), 25, 100).round(1)

    # Calculate weighted academic score + slight noise to simulate real-world risk
    composite_score = (
        0.30 * attendance +
        0.30 * internal_marks +
        0.20 * assignment_scores +
        0.20 * prev_sem_marks +
        np.random.normal(0, 4, num_samples)
    )

    # Students with composite score < 62 are labeled as At Risk (1), else Safe (0)
    at_risk = (composite_score < 62).astype(int)

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Attendance": attendance,
        "Internal_Marks": internal_marks,
        "Assignment_Scores": assignment_scores,
        "Previous_Sem_Marks": prev_sem_marks,
        "At_Risk": at_risk
    })

    # Ensure Student 102 matches the exact problem specification example (High Risk)
    df.loc[df["Student_ID"] == 102, [
        "Attendance", "Internal_Marks", "Assignment_Scores", "Previous_Sem_Marks", "At_Risk"
    ]] = [52.0, 41.0, 45.0, 48.0, 1]

    # Simulate missing values (~2%) to demonstrate data cleaning pipeline
    for col in ["Attendance", "Internal_Marks", "Assignment_Scores", "Previous_Sem_Marks"]:
        mask = np.random.rand(num_samples) < 0.02
        df.loc[mask & (df["Student_ID"] != 102), col] = np.nan

    # Impute missing values using median strategy
    feature_cols = ["Attendance", "Internal_Marks", "Assignment_Scores", "Previous_Sem_Marks"]
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols]).round(1)

    df.to_csv(filepath, index=False)
    print(f"[INFO] Dataset saved to '{filepath}' with {len(df)} student records.")
    return df, feature_cols


# ---------------------------------------------------------
# 2. MODEL TRAINING & EVALUATION (RANDOM FOREST)
# ---------------------------------------------------------
def train_and_evaluate():
    df, feature_cols = generate_and_prepare_data()

    X = df[feature_cols]
    y = df["At_Risk"]

    # 80/20 Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    feature_importances = dict(
        zip(feature_cols, (rf_model.feature_importances_ * 100).round(2))
    )

    print("\n================ MODEL EVALUATION REPORT ================")
    print(f"Algorithm          : Random Forest Classifier")
    print(f"Accuracy           : {acc * 100:.2f}%")
    print(f"Precision          : {prec * 100:.2f}%")
    print(f"Recall (Sensitivity): {rec * 100:.2f}%")
    print(f"F1-Score           : {f1 * 100:.2f}%")
    print(f"ROC-AUC Score      : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe (0)", "At Risk (1)"]))
    print("Feature Importances (%):", feature_importances)
    print("=========================================================\n")

    # Test sample output for Student 102
    s102 = df[df["Student_ID"] == 102][feature_cols]
    s102_prob = rf_model.predict_proba(s102)[0][1]
    risk_level = "HIGH" if s102_prob >= 0.70 else ("MEDIUM" if s102_prob >= 0.40 else "LOW")
    pred_result = "At Risk" if s102_prob >= 0.40 else "Safe"

    print("--- Sample Prediction Verification ---")
    print(f"Student: 102")
    print(f"Risk Level: {risk_level}")
    print(f"Predicted Result: {pred_result}\n")

    # Save trained model
    joblib.dump(rf_model, "rf_model.pkl")

    # Save evaluation metrics for Flask Dashboard display
    metrics_data = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm.tolist(),
        "feature_importances": feature_importances
    }
    with open("model_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    print("[INFO] Model saved to 'rf_model.pkl' and metrics saved to 'model_metrics.json'.")

if __name__ == "__main__":
    train_and_evaluate()