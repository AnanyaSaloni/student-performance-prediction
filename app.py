import pandas as pd
import itertools
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("StudentsPerformance.csv")
data["final_score"] = (data["math score"] + data["reading score"] + data["writing score"]) / 3

# Drop lunch and parental education columns entirely
data = data.drop(columns=["lunch", "parental level of education"])

data_enc = pd.get_dummies(data, drop_first=True)
X = data_enc.drop("final_score", axis=1)
y = data_enc["final_score"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# TERMINAL LOGS
# -----------------------------
print("\n=== TERMINAL SUMMARY ===")
print("Model Accuracy:", round(model.score(X_test, y_test), 4))

# Feature importance
importances = model.feature_importances_
sorted_idx = importances.argsort()[::-1][:5]
print("\nTop 5 Important Features:")
for i in sorted_idx:
    print(f" - {X.columns[i]}: {importances[i]:.3f}")

# Clustering summary
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
data["cluster"] = clusters
print("\nCluster Profiles (mean scores):")
print(data.groupby("cluster")[["math score","reading score","writing score","final_score"]].mean())

# -----------------------------
# INTERVENTION ENGINE
# -----------------------------
def apply_action(student, action):
    s = student.copy()
    if action == "test_prep":
        s["test preparation course"] = "completed"
    elif action == "engagement_boost":
        s["raisedhands"] = s.get("raisedhands", 0) + 10
    return s

def encode(student):
    df = pd.DataFrame([student])
    return pd.get_dummies(df).reindex(columns=X.columns, fill_value=0)

def get_best_path(student):
    actions = ["test_prep", "engagement_boost"]
    base_score = model.predict(encode(student))[0]
    best_score = base_score
    best_path = []
    for r in range(1, len(actions)+1):
        for combo in itertools.combinations(actions, r):
            temp = student.copy()
            for action in combo:
                temp = apply_action(temp, action)
            score = model.predict(encode(temp))[0]
            if score > best_score:
                best_score = score
                best_path = combo
    return base_score, best_score, best_path

# -----------------------------
# STREAMLIT DASHBOARD
# -----------------------------
st.set_page_config(page_title="Smart Student Performance Analysis and Intervention Recommendation System", layout="wide")

st.title("🎓 Smart Student Performance Analysis and Intervention Recommendation System")

tab1, tab2, tab3, tab4 = st.tabs(["Results", "Explainability", "Fairness Audit", "Cluster Profiles"])

with tab1:
    st.header("Results")
    gender = st.selectbox("Gender", ["male", "female"])
    race = st.selectbox("Class", ["group A","group B","group C","group D","group E"])
    prep = st.selectbox("Test Prep", ["none","completed"])
    math = st.slider("Math Score", 0, 100, 50)
    reading = st.slider("Reading Score", 0, 100, 45)
    writing = st.slider("Writing Score", 0, 100, 48)

    student = {
        "gender": gender,
        "race/ethnicity": race,
        "test preparation course": prep,
        "math score": math,
        "reading score": reading,
        "writing score": writing
    }

    if st.button("Analyze"):
        before, after, path = get_best_path(student)
        st.write("Before:", round(before,2))
        st.write("After:", round(after,2))
        st.write("Improvement:", round(after-before,2))
        fig, ax = plt.subplots()
        ax.bar(["Before","After"], [before, after], color=["red","green"])
        st.pyplot(fig)
        st.write("Best Intervention Path:", " → ".join(path) if path else "No effective intervention found")

with tab2:
    st.header("Explainability (Top Features)")
    fig, ax = plt.subplots()
    ax.barh([X.columns[i] for i in sorted_idx], importances[sorted_idx])
    ax.set_xlabel("Importance")
    st.pyplot(fig)

with tab3:
    st.header("Fairness Audit")
    st.write("Checking if model accuracy is consistent across groups...")
    for feature in ["gender_male","race/ethnicity_group B"]:
        if feature in X.columns:
            mask = X_test[feature] == 1
            if mask.any():
                acc = model.score(X_test[mask], y_test[mask])
                st.write(f"{feature}: Accuracy = {acc:.2f}")
            else:
                st.write(f"{feature}: Not enough data")

with tab4:
    st.header("Cluster Profiles")
    st.write("Silhouette Score:", silhouette_score(X, clusters))
    st.write(data.groupby("cluster")[["math score","reading score","writing score","final_score"]].mean())












