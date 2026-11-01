import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

FEATURES = ["Attendance", "Internal_Marks", "Assignment_Scores", "Previous_Sem_Marks"]

def load_artifacts():
    if not os.path.exists("rf_model.pkl") or not os.path.exists("student_data.csv"):
        from train_model import train_and_evaluate
        train_and_evaluate()

    model = joblib.load("rf_model.pkl")
    df = pd.read_csv("student_data.csv")
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
    return model, df, metrics

def classify_risk(prob):
    """Maps Random Forest probability of being at-risk to Risk Level and Predicted Result."""
    if prob >= 0.70:
        return "HIGH", "At Risk", "danger"
    elif prob >= 0.40:
        return "MEDIUM", "At Risk", "warning"
    else:
        return "LOW", "Safe", "success"

@app.route("/", methods=["GET", "POST"])
def dashboard():
    model, df, metrics = load_artifacts()

    # Run batch prediction across all students for the dashboard table
    probs = model.predict_proba(df[FEATURES])[:, 1]
    df["Risk_Prob"] = (probs * 100).round(1)

    risk_levels, pred_results, badge_colors = [], [], []
    for p in probs:
        lvl, res, color = classify_risk(p)
        risk_levels.append(lvl)
        pred_results.append(res)
        badge_colors.append(color)

    df["Risk_Level"] = risk_levels
    df["Predicted_Result"] = pred_results
    df["Badge_Color"] = badge_colors

    # Default spotlight prediction (Student 102)
    s102_row = df[df["Student_ID"] == 102].iloc[0]
    prediction_output = {
        "student_id": int(s102_row["Student_ID"]),
        "attendance": s102_row["Attendance"],
        "internal_marks": s102_row["Internal_Marks"],
        "assignment_scores": s102_row["Assignment_Scores"],
        "prev_sem_marks": s102_row["Previous_Sem_Marks"],
        "risk_level": s102_row["Risk_Level"],
        "predicted_result": s102_row["Predicted_Result"],
        "probability": s102_row["Risk_Prob"],
        "badge_color": s102_row["Badge_Color"]
    }

    # Handle manual prediction form submission
    if request.method == "POST":
        student_id = request.form.get("student_id", "Custom")
        attendance = float(request.form.get("attendance", 0))
        internal_marks = float(request.form.get("internal_marks", 0))
        assignment_scores = float(request.form.get("assignment_scores", 0))
        prev_sem_marks = float(request.form.get("prev_sem_marks", 0))

        input_df = pd.DataFrame([[
            attendance, internal_marks, assignment_scores, prev_sem_marks
        ]], columns=FEATURES)

        prob = model.predict_proba(input_df)[0][1]
        risk_level, predicted_result, badge_color = classify_risk(prob)

        prediction_output = {
            "student_id": student_id,
            "attendance": attendance,
            "internal_marks": internal_marks,
            "assignment_scores": assignment_scores,
            "prev_sem_marks": prev_sem_marks,
            "risk_level": risk_level,
            "predicted_result": predicted_result,
            "probability": round(prob * 100, 1),
            "badge_color": badge_color
        }

    # Summary statistics for KPI cards
    summary = {
        "total_students": len(df),
        "high_risk": int((df["Risk_Level"] == "HIGH").sum()),
        "medium_risk": int((df["Risk_Level"] == "MEDIUM").sum()),
        "low_risk": int((df["Risk_Level"] == "LOW").sum()),
    }

    # Display top 100 students sorted by highest risk first (keeping 102 near top)
    students_list = df.sort_values(by="Risk_Prob", ascending=False).head(100).to_dict(orient="records")

    return render_template(
        "index.html",
        prediction=prediction_output,
        summary=summary,
        metrics=metrics,
        students=students_list
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for external integrations."""
    model, _, _ = load_artifacts()
    data = request.get_json()

    input_df = pd.DataFrame([[
        float(data["attendance"]),
        float(data["internal_marks"]),
        float(data["assignment_scores"]),
        float(data["prev_sem_marks"])
    ]], columns=FEATURES)

    prob = model.predict_proba(input_df)[0][1]
    risk_level, predicted_result, _ = classify_risk(prob)

    return jsonify({
        "Student": data.get("student_id", "Unknown"),
        "Risk Level": risk_level,
        "Predicted Result": predicted_result,
        "Risk Probability (%)": round(prob * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)