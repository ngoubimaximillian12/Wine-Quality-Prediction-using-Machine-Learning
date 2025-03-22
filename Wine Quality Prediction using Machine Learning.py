import os
import zipfile
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sdv.single_table import CTGANSynthesizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import subprocess
import json
import base64
import datetime
import random

# Streamlit config
st.set_page_config(page_title="🍷 Wine AI App", layout="wide")

# Session state
if 'user_role' not in st.session_state:
    st.session_state.user_role = "Guest"
if 'favorite_wines' not in st.session_state:
    st.session_state.favorite_wines = []
if 'history_log' not in st.session_state:
    st.session_state.history_log = []

# Navigation based on user role
if st.session_state.user_role == "Admin":
    page = st.sidebar.selectbox("📚 Navigation", [
        "🏠 Home", "🍷 Predict Wine", "📦 Bulk Upload", "🧬 Generator", "📊 Dashboard",
        "🍽️ Wine Finder", "📥 Reports", "📒 History Log", "🤖 Taste Pairing AI", "🛠️ Admin Panel", "💬 Wine Chatbot"
    ])
elif st.session_state.user_role == "Consumer":
    page = st.sidebar.selectbox("📚 Navigation",
                                ["🏠 Home", "🍽️ Wine Finder", "🤖 Taste Pairing AI", "⭐ Favorites", "💬 Wine Chatbot"])
else:
    page = st.sidebar.selectbox("📚 Navigation", ["🏠 Home", "🔐 Login Panel"])

# Login Panel
if page == "🔐 Login Panel":
    st.title("🔐 Login Panel")
    username = st.text_input("Username")
    role = st.selectbox("Select Role", ["Admin", "Consumer"])
    if st.button("Login") and username:
        st.session_state.user_role = role
        st.experimental_rerun()
    st.stop()

# Kaggle download and unzip
kaggle_api = {"username": "ngoubimaximillian", "key": "0c31e26ac769f4ba290938601af046f0"}
with open("kaggle.json", "w") as f:
    json.dump(kaggle_api, f)

os.environ['KAGGLE_USERNAME'] = kaggle_api['username']
os.environ['KAGGLE_KEY'] = kaggle_api['key']

subprocess.run([
    "/Users/ngoubimaximilliandiamgha/Desktop/PythonProject12/.venv/bin/kaggle",
    "datasets", "download", "-d", "yasserh/wine-quality-dataset", "--force"
])

# Unzip the dataset
zip_path = "wine-quality-dataset.zip"
extract_dir = "wine_quality_data"
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(extract_dir, "WineQT.csv"))
    df['is_high_quality'] = (df['quality'] >= 7).astype(int)
    df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 1e-6)
    df['alcohol_density_ratio'] = df['alcohol'] / df['density']
    df['interaction_alc_acid'] = df['alcohol'] * df['citric acid']
    return df


df = load_data()
features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    'acidity_ratio', 'alcohol_density_ratio', 'interaction_alc_acid'
]


# Train model
@st.cache_resource
def train_model():
    X = df[features]
    y = df['is_high_quality']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "wine_quality_model.pkl")
    return model


model = train_model()

# Summary statistics for the data
summary_df = df.groupby('quality').agg({
    'alcohol': 'mean', 'pH': 'mean', 'residual sugar': 'mean',
    'density': 'mean', 'volatile acidity': 'mean', 'sulphates': 'mean'
}).reset_index()


# SHAP Explainer for Model Interpretability
@st.cache_resource
def create_shap_explainer(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer


# Home page
if page == "🏠 Home":
    st.title("🍷 Welcome to Wine Quality Intelligence App")
    st.markdown("""
        This application uses real wine chemical data and AI models to:
        - ✅ Predict wine quality from input features
        - 📈 Analyze and visualize wine quality trends
        - 🍽️ Recommend wine profiles for different consumer tastes
        - 📦 Allow admins to upload, manage, and analyze batch data
        - 🧬 Generate synthetic data for research and testing
    """)

    st.subheader("📊 Dataset Statistics Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Wines", df.shape[0])
        st.metric("Avg Alcohol", round(df['alcohol'].mean(), 2))
        st.metric("High Quality Wines", df['is_high_quality'].sum())
    with col2:
        st.metric("Avg pH", round(df['pH'].mean(), 2))
        st.metric("Avg Residual Sugar", round(df['residual sugar'].mean(), 2))
        st.metric("Unique Qualities", df['quality'].nunique())

    # Visualizations
    st.subheader("📈 Alcohol vs Wine Quality")
    fig1 = px.box(df, x="quality", y="alcohol", color="quality", title="Alcohol Content by Wine Quality", height=400)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🧪 Correlation Matrix (Features)")
    corr = df[features].corr()
    fig2 = px.imshow(corr, text_auto=True, title="Correlation Heatmap", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔬 Quality Score Distribution")
    fig3 = px.histogram(df, x="quality", nbins=7, color="quality", title="Distribution of Wine Quality Scores",
                        height=400)
    st.plotly_chart(fig3, use_container_width=True)

# Wine Finder (Feature Input for Prediction)
if page == "🍽️ Wine Finder":
    st.title("🍽️ Wine Finder")
    alcohol = st.slider("Alcohol %", float(df['alcohol'].min()), float(df['alcohol'].max()), 10.0)
    acidity = st.slider("Volatile Acidity", float(df['volatile acidity'].min()), float(df['volatile acidity'].max()),
                        0.5)
    sugar = st.slider("Residual Sugar", float(df['residual sugar'].min()), float(df['residual sugar'].max()), 5.0)

    # Filter wines based on input attributes
    results = df[(df['alcohol'] >= alcohol) & (df['volatile acidity'] <= acidity) & (df['residual sugar'] >= sugar)]
    st.write(f"🔍 Found {results.shape[0]} matching wines")
    st.dataframe(results.head(10))

# Wine Prediction (Interactive Prediction)
if page == "🍷 Predict Wine":
    st.title("🍷 Predict Wine Quality")

    # User Inputs
    alcohol_input = st.slider("Alcohol %", 0.0, 20.0, 10.0)
    acidity_input = st.slider("Volatile Acidity", 0.0, 2.0, 0.5)
    sugar_input = st.slider("Residual Sugar", 0.0, 50.0, 5.0)

    # Show prediction for given features
    input_features = np.array([alcohol_input, acidity_input, sugar_input]).reshape(1, -1)
    prediction = model.predict(input_features)
    prob = model.predict_proba(input_features)  # Predict probability

    # Show prediction and confidence
    st.write(f"📊 Predicted Quality: {prediction[0]} (1 = High Quality, 0 = Low Quality)")
    st.write(f"Confidence: {prob[0][1] * 100:.2f}%")

    # Provide insights based on feature importance
    shap_values, explainer = create_shap_explainer(model, df[features])
    shap.initjs()

    st.subheader("🌟 Feature Importance")
    shap.summary_plot(shap_values[1], df[features], plot_type="bar")  # Explanation of high-quality prediction

    # Statistical Insights
    st.subheader("📊 Statistical Insights for Wine Features")
    alcohol_stat = df['alcohol'].describe()
    acidity_stat = df['volatile acidity'].describe()
    sugar_stat = df['residual sugar'].describe()

    st.write(f"**Alcohol Content Statistics:**")
    st.write(alcohol_stat)
    st.write(f"**Volatile Acidity Statistics:**")
    st.write(acidity_stat)
    st.write(f"**Residual Sugar Statistics:**")
    st.write(sugar_stat)

    # Actionable insights
    st.subheader("💡 Actionable Insights")
    if alcohol_input < df['alcohol'].mean():
        st.warning("Increasing alcohol content can improve wine quality based on historical data.")
    if acidity_input > df['volatile acidity'].mean():
        st.warning("High acidity levels may lower the overall quality of the wine.")

# Bulk Upload Page (Unchanged)
if page == "📦 Bulk Upload":
    st.title("📦 Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        batch_df['prediction'] = model.predict(batch_df[features])
        batch_df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history_log.append(batch_df)
        st.success("✅ Predictions complete!")
        st.dataframe(batch_df)
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "wine_predictions.csv")

# History Log Page (Unchanged)
if page == "📒 History Log":
    st.title("📒 Prediction History Log")
    if st.session_state.history_log:
        for entry in st.session_state.history_log:
            st.dataframe(entry)
    else:
        st.info("No predictions yet.")

# Wine Similarity Tool (Find Similar Wines)
if page == "🍽️ Wine Finder":
    st.title("🍽️ Wine Similarity")
    alcohol = st.slider("Alcohol %", float(df['alcohol'].min()), float(df['alcohol'].max()), 10.0)
    acidity = st.slider("Volatile Acidity", float(df['volatile acidity'].min()), float(df['volatile acidity'].max()),
                        0.5)
    sugar = st.slider("Residual Sugar", float(df['residual sugar'].min()), float(df['residual sugar'].max()), 5.0)

    # Calculate similarity (Euclidean distance)
    df['similarity'] = np.sqrt(
        (df['alcohol'] - alcohol) ** 2 + (df['volatile acidity'] - acidity) ** 2 + (df['residual sugar'] - sugar) ** 2
    )

    # Find most similar wines
    similar_wines = df.nsmallest(5, 'similarity')
    st.write(f"🔍 Found {similar_wines.shape[0]} most similar wines:")
    st.dataframe(similar_wines[['quality', 'alcohol', 'volatile acidity', 'residual sugar', 'similarity']])

# Wine Attribute Clustering (KMeans)
if page == "📊 Dashboard":
    st.title("📊 Wine Clustering Dashboard")

    # Use KMeans clustering to group wines
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[features])

    # Visualize clusters
    fig = px.scatter(df, x='alcohol', y='volatile acidity', color='cluster', title="Wine Clusters based on Features")
    st.plotly_chart(fig)

