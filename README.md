🎓 Smart Student Performance Analysis and Intervention Recommendation System

📌 Overview
This project predicts student performance using academic and behavioral datasets and recommends unique interventions to improve outcomes. It was developed as part of an AI/ML internship project.

The system combines:
Academic scores (math, reading, writing)
Behavioral engagement features (raised hands, resources visited, discussions)
Machine learning models (Random Forest + clustering)
Streamlit dashboard with tabs for Results, Explainability, Fairness Audit, and Cluster Profiles

⚙️ Tech Stack
Python (pandas, numpy, scikit‑learn, matplotlib)
Streamlit (interactive dashboard)
GitHub (version control, portfolio hosting)

🚀 Features
Results Tab: Predicts student performance and shows before/after intervention improvements
Explainability Tab: Displays top feature importance for transparency
Fairness Audit Tab: Evaluates model accuracy across gender and class groups
Cluster Profiles Tab: Groups students into clusters with average scores for peer‑based insights

📊 Outputs

Model Training

python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

Intervention Engine

python
def get_best_path(student):
    actions = ["test_prep", "engagement_boost"]
    base_score = model.predict(encode(student))[0]
    # simulate interventions
    ...
    return base_score, best_score, best_path


Model accuracy and top features (terminal logs)
Streamlit dashboard with interactive tabs
Intervention engine with unique actions (test prep, engagement boost)
Cluster profiles with silhouette score
