# AI-Based Student Risk Prediction — Presentation Notes

## 1. Project title

**AI-Based Student Risk Prediction and Early Warning Dashboard**

One-line description: This project uses a machine-learning classifier to identify students who may be academically at risk from attendance and assessment performance, then presents the results through a Flask web dashboard.

Suggested duration: 8–12 minutes presentation and 3–5 minutes live demonstration.

---

## 2. Opening script

“Good morning. My project is an AI-Based Student Risk Prediction Dashboard. The purpose is to identify students who may be at risk before their final result, so that teachers can provide timely support. The system accepts four academic indicators—attendance, internal marks, assignment scores, and previous-semester marks. A Random Forest model calculates the probability of risk, classifies the student as LOW, MEDIUM, or HIGH risk, and displays the result in a web dashboard. The project also provides model evaluation metrics, feature importance, student filtering, and a REST API for integration.”

---

## 3. Problem statement

Traditional identification of weak students is often manual and may happen only after examination results. This can delay intervention.

The project addresses this by answering:

> “Given a student’s attendance and academic indicators, is the student likely to be at risk, and how confident is the prediction?”

The output is intended as an early-warning aid for teachers, not as a final or permanent judgment about a student.

---

## 4. Objectives

- Generate or use structured student academic data.
- Clean missing feature values using median imputation.
- Train a supervised binary classifier: Safe or At Risk.
- Evaluate the classifier using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Convert model probability into LOW, MEDIUM, and HIGH risk levels.
- Provide an interactive web dashboard for individual and batch analysis.
- Expose a `/api/predict` endpoint for external applications.

---

## 5. Technology stack

- **Python:** data processing and machine learning.
- **Pandas and NumPy:** tabular data handling and numerical generation.
- **Scikit-learn:** imputation, train/test split, Random Forest, and evaluation metrics.
- **Joblib:** saving and loading the trained model.
- **Flask:** backend web application and REST API.
- **HTML, Bootstrap, JavaScript, Chart.js:** dashboard interface and charts.
- **CSV and JSON:** dataset and saved evaluation metrics.

Run commands for the demonstration:

```text
pip install -r requirements.txt
python train_model.py       # only if the model must be regenerated
python app.py
```

Open `http://127.0.0.1:5000/` in a browser.

---

## 6. Project files to explain

| File | Purpose |
|---|---|
| `train_model.py` | Generates/prepares data, trains Random Forest, evaluates it, and saves artifacts. |
| `app.py` | Flask routes, prediction logic, dashboard data preparation, and API endpoint. |
| `templates/index.html` | Dashboard layout, form, charts, roster table, and JavaScript filters. |
| `student_data.csv` | 500 student records and the binary `At_Risk` target. |
| `rf_model.pkl` | Serialized trained Random Forest model. |
| `model_metrics.json` | Saved evaluation metrics used by the dashboard. |
| `requirements.txt` | Required Python libraries. |

---

## 7. Data and preprocessing explanation

The current dataset contains **500 students**, with IDs from 101 onward. The four input features are:

1. Attendance percentage
2. Internal marks
3. Assignment scores
4. Previous-semester marks

The target is `At_Risk`: `1` means At Risk and `0` means Safe.

In `train_model.py`, the data is generated with a fixed NumPy seed of 42, making the experiment reproducible. Values are clipped to realistic ranges. A weighted composite score is used to create the training label:

```text
composite score = 0.30 attendance
                + 0.30 internal marks
                + 0.20 assignment scores
                + 0.20 previous-semester marks
                + noise
```

Students with a composite score below 62 are labeled At Risk. Approximately 2% missing values are introduced to demonstrate preprocessing, then replaced using the median of each feature. Student 102 is deliberately set to a low-performing example for the demonstration.

Important limitation to state honestly: the included dataset is synthetic/demo data. For a production system, labels should come from real historical outcomes, with privacy protection and institutional validation.

---

## 8. Model and training explanation

The project uses a **Random Forest Classifier**, an ensemble of decision trees. Each tree learns different splits of the academic features, and the forest combines their predictions. It is useful here because it can model non-linear relationships, works well on tabular data, and provides feature importance values.

Training configuration:

- 150 trees (`n_estimators=150`)
- Maximum tree depth of 8
- Minimum samples to split a node: 4
- `class_weight="balanced"` to reduce the effect of class imbalance
- 80% training data and 20% test data
- Stratified split so the class proportions are preserved
- Fixed `random_state=42` for reproducibility

The model outputs the probability of class 1, which is the probability that a student is At Risk.

Risk mapping in `app.py`:

- Probability `< 0.40`: **LOW**, result **Safe**
- Probability `0.40–<0.70`: **MEDIUM**, result **At Risk**
- Probability `>= 0.70`: **HIGH**, result **At Risk**

These thresholds are application rules layered on top of the model probability; they are not additional machine-learning classes.

---

## 9. Model results to present

The saved test-set results are:

| Metric | Value | Meaning |
|---|---:|---|
| Accuracy | 91.00% | Correct predictions among all test cases. |
| Precision | 66.67% | Of predicted at-risk students, the proportion actually at risk. |
| Recall | 61.54% | Of actual at-risk students, the proportion detected. |
| F1-score | 64.00% | Balance between precision and recall. |
| ROC-AUC | 0.916 | Overall ranking ability across thresholds. |

Confusion matrix, in the order Safe / At Risk:

```text
[[83, 4],
 [ 5, 8]]
```

Interpretation: 83 safe students were correctly identified, 4 safe students were incorrectly flagged, 5 at-risk students were missed, and 8 at-risk students were correctly detected.

Feature importance:

- Internal Marks: 36.61%
- Attendance: 27.08%
- Assignment Scores: 23.22%
- Previous Sem Marks: 13.09%

Explain that feature importance indicates how much the trained forest used a feature for splitting; it does not prove causation.

---

## 10. Live demonstration script

1. Start the Flask application and open the dashboard.
2. Point out the KPI cards: total students, high-risk count, medium-risk count, and test accuracy.
3. Explain the prediction form and its four model inputs.
4. Use the default Student 102 example:
   - Attendance: 52
   - Internal marks: 41
   - Assignment scores: 45
   - Previous-semester marks: 48
   - Expected result: approximately **99.1% risk probability, HIGH, At Risk**.
5. Change the values to a strong student, for example 90, 85, 88, and 84. Show that the probability should fall and the result should become Safe/LOW.
6. Show the feature-importance chart and explain that internal marks are the strongest feature in this trained model.
7. Demonstrate the roster: sort order is highest risk first; use the risk dropdown and student-ID search.
8. Click **Inspect** on a row to load its values into the prediction form.
9. If asked about integration, explain that `POST /api/predict` accepts JSON fields `attendance`, `internal_marks`, `assignment_scores`, `prev_sem_marks`, and optionally `student_id`.

Example API request body:

```json
{
  "student_id": "102",
  "attendance": 52,
  "internal_marks": 41,
  "assignment_scores": 45,
  "prev_sem_marks": 48
}
```

---

## 11. System flow to draw on the board

```text
Student CSV / form input
          ↓
Feature selection and median imputation
          ↓
Random Forest model
          ↓
At-risk probability
          ↓
LOW / MEDIUM / HIGH classification
          ↓
Dashboard result or JSON API response
```

---

## 12. Likely viva questions and answers

### Basic questions

**Q1. What is the aim of the project?**

To provide an early warning of academic risk using attendance and assessment data, allowing timely intervention.

**Q2. Is this classification or regression?**

Classification, because the target is categorical: Safe or At Risk. The model also returns a probability.

**Q3. What are the input features?**

Attendance, internal marks, assignment scores, and previous-semester marks.

**Q4. What is the target variable?**

`At_Risk`, where 1 is At Risk and 0 is Safe.

**Q5. Why use Random Forest?**

It performs well on tabular data, captures non-linear patterns, is less sensitive to individual noisy trees, and provides feature importance.

### Machine-learning questions

**Q6. Why split the data into training and testing sets?**

The training set learns the model, while the unseen test set estimates how well it generalizes.

**Q7. Why use a stratified split?**

To preserve the proportion of Safe and At Risk classes in both subsets.

**Q8. What is overfitting? How is it controlled here?**

Overfitting is memorizing training examples instead of learning general patterns. The project uses a test set, limited tree depth, minimum split size, and multiple trees. Cross-validation would strengthen the evaluation further.

**Q9. Why is recall important in this project?**

Recall measures how many truly at-risk students are detected. Missing such a student can delay support, so recall is important, although excessive false alarms also have a cost.

**Q10. Explain the confusion matrix.**

TN=83, FP=4, FN=5, and TP=8. The model correctly detected 8 at-risk students but missed 5 in the test set.

**Q11. What is the difference between precision and recall?**

Precision asks “when the model flags risk, how often is it correct?” Recall asks “of all genuinely at-risk students, how many did it find?”

**Q12. Why is accuracy alone insufficient?**

If classes are imbalanced, a model can achieve high accuracy by mostly predicting the majority class. Precision, recall, F1, confusion matrix, and ROC-AUC provide a fuller view.

**Q13. Why is median imputation used?**

It replaces missing numerical values with a robust central value and is less affected by extreme outliers than the mean.

**Q14. What does `class_weight="balanced"` do?**

It gives more influence to the minority class during training, helping the model pay attention to at-risk cases.

### Application questions

**Q15. How does Flask connect the model to the UI?**

Flask loads the serialized model, receives form data, creates a one-row DataFrame with the same feature order, calls `predict_proba`, applies the risk rules, and renders the Jinja template.

**Q16. What is the purpose of `rf_model.pkl`?**

It stores the trained model so the application can make predictions without retraining whenever the server starts.

**Q17. What does the API endpoint do?**

`POST /api/predict` accepts feature values as JSON and returns the student ID, risk level, predicted result, and risk probability.

**Q18. What happens when a user clicks Inspect?**

JavaScript copies that row’s values into the form and submits the form, triggering a new prediction.

**Q19. Why are there three risk levels if the model is binary?**

The model is binary, but its probability is grouped into three operational bands so staff can prioritize intervention.

**Q20. What are the limitations?**

The data is synthetic, labels are generated from a weighted rule, the test set is relatively small, the thresholds are hand-selected, and the model does not include socioeconomic, behavioral, or contextual information. It should support—not replace—human judgment.

**Q21. How would you improve the project?**

Use validated historical data, cross-validation, calibration, threshold tuning for higher recall, explainability such as SHAP, authentication, database storage, monitoring for drift, fairness testing, and role-based access.

---

## 13. Suggested marks/viva preparation

If the demonstration is graded out of 100, a defensible breakdown is:

| Area | Marks |
|---|---:|
| Problem definition and objectives | 10 |
| Data understanding and preprocessing | 15 |
| Algorithm selection and implementation | 20 |
| Evaluation and interpretation | 15 |
| Web application and live demonstration | 20 |
| Viva answers and limitations | 15 |
| Presentation clarity | 5 |
| **Total** | **100** |

To maximize marks, clearly explain the confusion matrix, demonstrate Student 102, show one safe example, mention the synthetic-data limitation, and distinguish probability thresholds from model classes.

## 14. Closing script

“To conclude, this project combines machine learning with a practical Flask dashboard for early identification of students who may need academic support. The Random Forest model uses four measurable indicators, produces a risk probability, and makes the result easy to inspect through charts and a searchable roster. The current implementation is a reproducible demonstration; the next step would be validating it on real institutional data with privacy, fairness, and monitoring controls. Thank you.”

