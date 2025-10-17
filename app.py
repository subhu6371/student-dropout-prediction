import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import base64
import matplotlib.pyplot as plt

# --- 🔹 Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="EarlyGuard | AI Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🔹 Utility Functions ---
def set_bg_local(image_file):
    """Set local background image"""
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}
            .main .block-container {{
                background: rgba(10, 5, 30, 0.75);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 2rem;
                border: 1px solid rgba(173, 216, 230, 0.2);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Background image not found. Please add a 'background.jpg' file to your project folder.")

def load_lottieurl(url: str):
    """Load a Lottie animation from a URL."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- 🔹 Load Model & Assets ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('ultimate_stacking_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure 'ultimate_stacking_model.pkl' and 'model_columns.pkl' are in the same folder.")
        return None, None

model, model_columns = load_assets()
if model is None or model_columns is None:
    st.stop()

# --- 🔹 Background & Animations ---
set_bg_local("background.jpg")

edu_anim = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
success_anim = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")
warning_anim = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_kxsd2ytq.json")

# --- 🔹 Sidebar Navigation ---
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🎓 Single Prediction", "📁 Batch Prediction"])

# ------------------------------
# 🏠 Home Page
# ------------------------------
if page == "🏠 Home":
    st.title("📘 Student Dropout Prediction Dashboard")
    st.markdown("An AI-powered system to predict and visualize *student dropout risks*, built with an advanced, API-enriched model.")

    if edu_anim:
        st_lottie(edu_anim, height=300, key="edu")

    st.markdown("---")
    st.subheader("Project Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🎓 Single Student Prediction")
        st.write("Predict dropout risk for an individual student with detailed analysis.")
    with col2:
        st.info("🌐 API Enriched Data")
        st.write("Our model is trained on student data enriched with real-world economic and environmental data.")
    with col3:
        st.warning("🧠 Advanced Modeling")
        st.write("We use a Stacking Ensemble model for the highest possible accuracy and recall.")

# ------------------------------
# 🎓 Single Student Prediction
# ------------------------------
elif page == "🎓 Single Prediction":
    st.header("🎓 Predict for a Single Student")

    with st.form("student_input_form"):
        inputs = {}
        col1, col2, col3 = st.columns(3)

        with col1:
            failures = st.slider("Number of Past Failures", 0, 4, 0)
            absences = st.slider("Number of Absences", 0, 93, 2)
            age = st.slider('Age', 15, 22, 17)
            sex = st.radio('Sex', ['Female', 'Male'])
            address = st.radio('Address', ['Rural', 'Urban'])
            
        with col2:
            Mjob = st.selectbox("Mother's Job", ['Teacher', 'Health', 'Services', 'At Home', 'Other'])
            Fjob = st.selectbox("Father's Job", ['Teacher', 'Health', 'Services', 'At Home', 'Other'])
            traveltime = st.slider('Travel Time', 1, 4, 2)
            studytime = st.slider('Study Time (hours/week)', 1, 4, 2)
            
        with col3:
            Medu = st.slider("Mother's Education", 0, 4, 2)
            Fedu = st.slider("Father's Education", 0, 4, 2)
            internet = st.radio('Has Internet Access?', ['Yes', 'No'])
            higher = st.radio('Wants Higher Education?', ['Yes', 'No'])

        submitted = st.form_submit_button("🔮 Predict Dropout Risk")

    if submitted:
        # --- Create the input DataFrame from UI controls ---
        input_data = {
            'failures': failures, 'absences': absences, 'age': age, 'traveltime': traveltime,
            'studytime': studytime, 'Medu': Medu, 'Fedu': Fedu
        }
        # One-hot encode categorical features
        input_data['sex_M'] = 1 if sex == 'Male' else 0
        input_data['address_U'] = 1 if address == 'Urban' else 0
        input_data['internet_yes'] = 1 if internet == 'Yes' else 0
        input_data['higher_yes'] = 1 if higher == 'Yes' else 0
        input_data['Mjob_teacher'] = 1 if Mjob == 'Teacher' else 0; input_data['Mjob_health'] = 1 if Mjob == 'Health' else 0; input_data['Mjob_services'] = 1 if Mjob == 'Services' else 0; input_data['Mjob_other'] = 1 if Mjob == 'Other' else 0
        input_data['Fjob_teacher'] = 1 if Fjob == 'Teacher' else 0; input_data['Fjob_health'] = 1 if Fjob == 'Health' else 0; input_data['Fjob_services'] = 1 if Fjob == 'Services' else 0; input_data['Fjob_other'] = 1 if Fjob == 'Other' else 0
        
        # Add the fixed API data we discovered
        input_data['gdp_per_capita'] = 19215.78
        input_data['poor_air_quality_days'] = 0
        input_data['exam_stress_score'] = 45.3
        input_data['avg_education_sentiment'] = 0.0

        input_df = pd.DataFrame([input_data])
        
        # --- Align the DataFrame to match the model's training data ---
        final_df = pd.DataFrame(columns=model_columns)
        final_df = pd.concat([final_df, input_df], ignore_index=True).fillna(0)
        final_df = final_df[model_columns]
        
        # --- Make Prediction ---
        prediction = model.predict(final_df)[0]
        probability = model.predict_proba(final_df)[0][1] * 100

        # --- Display Results ---
        st.markdown("---")
        st.header("📈 Prediction Result")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if prediction == 1:
                st.error(f"⚠ High Dropout Risk: *{probability:.2f}%*")
                if warning_anim: st_lottie(warning_anim, height=180, key="warn")
            else:
                st.success(f"✅ Low Dropout Risk: *{probability:.2f}%*")
                if success_anim: st_lottie(success_anim, height=180, key="safe")
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                title={'text': "Risk Gauge (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#FF4B4B" if prediction == 1 else "#00FF7F"}}
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10), font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

        # --- Analysis Section ---
        st.markdown("---")
        st.header("🧠 Prediction Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        with analysis_col1:
            st.subheader("📊 Profile Comparison")
            avg_data = {
                'Metric': ['Past Failures', 'Absences'],
                'This Student': [failures, absences],
                'Average At-Risk Student': [1.5, 12]
            }
            comp_df = pd.DataFrame(avg_data).set_index('Metric')
            st.bar_chart(comp_df)
        
        with analysis_col2:
            st.subheader("🎯 Student Focus Card")
            focus_points = []
            if studytime < 2: focus_points.append("📚 Increase weekly study time.")
            if failures > 0: focus_points.append("📝 Review past mistakes and seek help in weak areas.")
            if absences > 10: focus_points.append("🏫 Improve class attendance.")
            if not focus_points: focus_points.append("🌟 Great work! Keep up the consistent effort.")
            for point in focus_points: st.info(point)

# ------------------------------
# 📁 Batch Prediction
# ------------------------------
elif page == "📁 Batch Prediction":
    st.header("📁 Predict for Multiple Students")
    st.warning("This feature is under development. Please use the Single Student prediction for now.")