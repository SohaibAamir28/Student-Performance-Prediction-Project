import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Page configuration
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e445e;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
Predicting the **Performance Index** based on academic and lifestyle factors.
Using Multiple Linear Regression.

**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)  
**GitHub:** [Student-Performance-Prediction-Project](https://github.com/SohaibAamir28/Student-Performance-Prediction-Project)
""")

# Paths
DATA_PATH = "dataset/Student_Performance.csv"
MODEL_PATH = "applied-ml/models/student_performance_model.pkl"

if os.path.exists(DATA_PATH) and os.path.exists(MODEL_PATH):
    # Load Model
    model = joblib.load(MODEL_PATH)
    
    # Load Data for Visuals
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    df_clean = df.copy()
    df_clean['Extracurricular Activities'] = df_clean['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    
    # Features and Target
    features = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 
                'Sleep Hours', 'Sample Question Papers Practiced']
    target = 'Performance Index'

    # Calculate Metrics on the fly for the dashboard
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_true = df_clean[target]
    y_pred_all = model.predict(df_clean[features])
    
    r2 = r2_score(y_true, y_pred_all)
    n = len(y_true)
    p = len(features)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_all))
    mae = mean_absolute_error(y_true, y_pred_all)

    # Metrics Layout
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{r2:.4f}")
    m2.metric("Adj R² Score", f"{adj_r2:.4f}")
    m3.metric("RMSE", f"{rmse:.2f}")
    m4.metric("MAE", f"{mae:.2f}")

    st.divider()

    # Sidebar for Predictions
    st.sidebar.header("🔍 Predict Score")
    
    hours_studied = st.sidebar.slider("Hours Studied", 1, 9, 5)
    prev_scores = st.sidebar.slider("Previous Scores", 40, 99, 70)
    extra_activities = st.sidebar.selectbox("Extracurricular Activities", options=["Yes", "No"], index=1)
    sleep_hours = st.sidebar.slider("Sleep Hours", 4, 9, 7)
    papers_practiced = st.sidebar.slider("Sample Question Papers Practiced", 0, 9, 5)
    
    extra_val = 1 if extra_activities == "Yes" else 0
    
    input_data = pd.DataFrame([[hours_studied, prev_scores, extra_val, sleep_hours, papers_practiced]], 
                              columns=features)
    
    if st.sidebar.button("Predict Performance Index"):
        pred = model.predict(input_data)[0]
        st.sidebar.success(f"Predicted Score: **{pred:.2f}**")
        st.sidebar.progress(min(max(int(pred), 0), 100))

    # Main Visuals
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Heatmap")
        corr = df_clean[features + [target]].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col2:
        st.subheader("Feature Importance")
        coef_df = pd.DataFrame({'Feature': features, 'Weight': model.coef_})
        fig_coef = px.bar(coef_df, x='Feature', y='Weight', color='Weight', color_continuous_scale='Tealgrn')
        st.plotly_chart(fig_coef, use_container_width=True)

    st.subheader("Actual vs Predicted (Full Dataset)")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_true, y=y_pred_all, mode='markers', name='Predictions', marker=dict(opacity=0.5)))
    fig_scatter.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], 
                                     y=[y_true.min(), y_true.max()], 
                                     mode='lines', name='Ideal Line', line=dict(color='red', dash='dash')))
    fig_scatter.update_layout(xaxis_title="Actual Score", yaxis_title="Predicted Score")
    st.plotly_chart(fig_scatter, use_container_width=True)

else:
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}")
    if not os.path.exists(MODEL_PATH):
        st.warning("Model file not found. Please run the training script first.")
