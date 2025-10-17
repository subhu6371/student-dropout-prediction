# 🧠 EarlyGuard | Student Dropout Prediction System

A machine learning web app that predicts the probability of student dropouts based on academic and socio-economic factors.

## 🚀 Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit (Web App)
- Joblib, SHAP, Matplotlib

## 📊 Features
- Predicts student dropout risk.
- Interactive visualization (SHAP-based model explanation).
- Clean and responsive Streamlit UI.

## 🧩 How It Works
1. Data preprocessing and model training in Jupyter (`notebooks/model_training.ipynb`).
2. Model saved using `joblib`.
3. Streamlit web app in `app.py` for real-time prediction.

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/student-dropout-prediction.git
cd student-dropout-prediction
pip install -r requirements.txt
streamlit run app.py
