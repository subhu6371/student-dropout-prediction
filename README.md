# AI-Powered Student Dropout Prediction

An interactive web application built with Streamlit and Scikit-learn to predict the risk of student dropout. This tool helps educational institutions identify at-risk students proactively, allowing for timely interventions.

![App Screenshot]("C:\Users\subha\OneDrive\Desktop\advanced_dropout_project\app-screenshot.png.png")
*Replace the link above with a screenshot of your app. You can upload the screenshot to your GitHub repo and link to it.*

---

## 📖 About The Project

The goal of this project is to leverage machine learning to address the critical issue of student attrition. By analyzing various academic and demographic factors, the model provides a probability score indicating a student's likelihood of dropping out. The web interface makes this powerful tool accessible to non-technical users like educators and administrators.

---

## ✨ Features

* **Single Student Prediction:** Enter a student's details through a user-friendly form to get an instant risk prediction.
* **Interactive Interface:** Built with Streamlit for a seamless and responsive user experience.
* **Machine Learning Backend:** Powered by a robust classification model (`ultimate_stacking_model.pkl`) trained on a comprehensive dataset.

---

## 🛠️ Built With

This project was built using the following technologies:

* **Backend:** Python
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Web Framework:** Streamlit
* **Data:** [UCI Machine Learning Repository - Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have Python 3.8+ and pip installed on your system.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/subhu6371/student-dropout-prediction.git](https://github.com/subhu6371/student-dropout-prediction.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd student-dropout-prediction
    ```
3.  **Install the required packages:**
    *(First, you should create a `requirements.txt` file by running `pip freeze > requirements.txt` in your project terminal and pushing it to GitHub)*
    ```sh
    pip install -r requirements.txt
    ```

### Usage

To run the Streamlit application, execute the following command in your terminal:

```sh
streamlit run app.py
