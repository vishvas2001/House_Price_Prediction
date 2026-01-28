# 🏠 Housing Price Prediction – Machine Learning Web App

An **end-to-end Machine Learning project** that predicts house prices using a trained **scikit-learn pipeline** and an interactive **Streamlit web application**.

This project focuses on **correct ML engineering practices**:
- feature-consistent inference,
- preprocessing with `ColumnTransformer`,
- model evaluation & explainability,
- and clean deployment without unnecessary backend complexity.

---

## 🚀 Project Overview

**Goal:** Predict house prices based on property features such as area, rooms, amenities, and furnishing status.

- **Problem Type:** Supervised Regression  
- **Target Variable:** `price`  
- **Interface:** Streamlit Web App  
- **Deployment Style:** Frontend + ML inference in one app  

---

## 🧠 Machine Learning Details

### 📊 Dataset
- Real-world housing dataset (`Housing.csv`)
- Combination of numerical & categorical features
- No missing values

### 🔢 Numerical Features
- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `parking`

### 🔤 Categorical Features
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `prefarea`
- `furnishingstatus`

---

## ⚙️ Model Architecture

- **Pipeline-based ML system (scikit-learn)**
- **Preprocessing** using `ColumnTransformer`
  - `StandardScaler` for numerical features
  - `OneHotEncoder` for categorical features
- **Model:** Ridge Regression
- **Serialization:** `joblib`

Using a pipeline ensures:
- No data leakage
- Identical preprocessing during training & inference
- Production-ready inference logic

---

## 📈 Model Evaluation

The app displays real performance metrics calculated from the dataset:

- **R² Score**
- **RMSE (Root Mean Squared Error)**

These metrics help users and reviewers understand model quality.

---

## 📊 Data Visualization

The Streamlit app includes:
- **Price distribution histogram** of the dataset
- Interactive UI to understand prediction context

---

## 🖥️ Streamlit Web Application

### UI Features
- Sliders for numerical inputs
- Dropdowns for categorical features
- Organized layout using sections & columns
- Real-time house price prediction
- Model performance metrics (R², RMSE)
- Dataset price distribution chart

---

## 📂 Project Structure

House_Price_Prediction/
│
├── streamlit_app/
│ ├── app.py # Streamlit application
│ ├── model.pkl # Trained ML pipeline
│ └── requirements.txt
│
├── ml/
│ ├── data/
│ │ └── Housing.csv
│ ├── train.py # Model training script
│ └── evaluate.py
│
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis
│
└── README.md


---

## ▶️ Run the Project Locally

### 1️⃣ Create & activate environment
```bash
python -m venv ml_env
ml_env\Scripts\activate
```