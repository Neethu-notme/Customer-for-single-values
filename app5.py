import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

st.set_page_config(page_title="Customer Segmentation Predictor", layout="centered")

st.title("🧠 Customer Cluster Prediction (KMeans)")
st.write("Predict which customer cluster a user belongs to")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("marketing_eda.csv")
    return df

df = load_data()

#fills empty cells
df_filled = df.fillna(df.mean(numeric_only=True))
# --------------------------------------------------
# Select features
# --------------------------------------------------
cat_features = ["Education", "Marital_Status"]
num_features = ["Income"]

df_model = df_filled[cat_features + num_features].dropna()

# --------------------------------------------------
# Encoding categorical variables
# --------------------------------------------------
df_encoded = pd.get_dummies(df_model, columns=cat_features, drop_first=True)

# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# --------------------------------------------------
# Train KMeans
# --------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# --------------------------------------------------
# SAVE MODEL & SCALER
# --------------------------------------------------
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
st.subheader("🔍 Enter Customer Details")

education = st.selectbox("Education", df["Education"].unique())
marital_status = st.selectbox("Marital Status", df["Marital_Status"].unique())
income = st.selectbox("Income", df["Income"].unique() )

# --------------------------------------------------
# Create input dataframe
# --------------------------------------------------
input_df = pd.DataFrame({
    "Education": [education],
    "Marital_Status": [marital_status],
    "Income": [income]
})

# One-hot encode input
input_encoded = pd.get_dummies(input_df, columns=cat_features, drop_first=True)

# Align columns with training data
input_encoded = input_encoded.reindex(columns=df_encoded.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_encoded)

# --------------------------------------------------
# Prediction + Confidence
# --------------------------------------------------
if st.button("🔮 Predict Cluster"):

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    # Distance to centroids
    distances = euclidean_distances(input_scaled, kmeans.cluster_centers_)[0]

    # Convert distance to confidence (inverse distance)
    confidence = 1 / distances
    probabilities = confidence / confidence.sum()

    st.success(f"✅ This customer belongs to **Cluster {cluster}**")

    # Cluster interpretation
    if cluster == 0:
        st.info("Cluster 0: Likely lower spending / conservative customers")
    else:
        st.info("Cluster 1: Likely higher spending / responsive customers")

    # --------------------------------------------------
    # Cluster Probability Visualization
    # --------------------------------------------------
    st.subheader("📊 Cluster Confidence")

    prob_df = pd.DataFrame({
        "Cluster": ["Cluster 0", "Cluster 1"],
        "Confidence": probabilities
    })

    st.bar_chart(prob_df.set_index("Cluster"))

    st.write(
        f"Prediction confidence for Cluster {cluster}: "
        f"**{probabilities[cluster]*100:.2f}%**"
    )




