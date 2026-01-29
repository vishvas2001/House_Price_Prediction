# 🏠 House Price Prediction – Machine Learning Web App

An end-to-end **Machine Learning web application** that predicts house prices using a trained **scikit-learn pipeline**, deployed with **Streamlit**.

🔗 **Live App:**  
https://house-price-prediction-ml-odel.streamlit.app

---

## 🚀 Project Overview

- **Problem Type:** Supervised Regression  
- **Target Variable:** `price`  
- **Interface:** Streamlit Web App  
- **Deployment:** Streamlit Community Cloud  

The project demonstrates correct **ML engineering practices** including feature-consistent inference, pipelines, evaluation metrics, and interactive visualization.

---

## 🧠 Machine Learning Details

### Dataset
- Real-world housing dataset (`Housing.csv`)
- Mix of numerical and categorical features
- No missing values

### Features

**Numerical**
- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `parking`

**Categorical**
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `prefarea`
- `furnishingstatus`

---

## ⚙️ Model Architecture

- **Pipeline-based model (scikit-learn)**
- Preprocessing with `ColumnTransformer`
  - `StandardScaler` for numerical features
  - `OneHotEncoder` for categorical features
- **Model:** Ridge Regression
- **Serialization:** `joblib`

Using a pipeline ensures the same preprocessing is applied during both training and inference.

---

## 📈 Model Evaluation (Displayed in App)

- **R² Score**
- **RMSE (Root Mean Squared Error)**

Metrics are calculated using the dataset to provide transparency about model performance.

---

## 📊 App Features

- Interactive sliders & dropdowns for inputs
- Real-time house price prediction
- Price distribution histogram of dataset
- Model performance metrics (R² & RMSE)
- Clean, responsive UI built with Streamlit

---

## 📂 Project Structure
```
House_Price_Prediction/
│
├── streamlit_app/
│ ├── app.py  # Streamlit application
│ ├── model.pkl # Trained ML pipeline
│ └── requirements.txt
│
├── ml/
│ ├── data/
│ │ └── Housing.csv
│ ├── train.py # Offline model training
│ └── evaluate.py # Offline evaluation
│
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis
│
└── README.md
```

---

## ▶️ Run Locally

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

## 🧠 Key Learnings
* Importance of feature schema consistency

* Proper use of ML pipelines for inference

* Handling categorical data correctly

* Evaluating regression models using RMSE & R²

* Deploying ML apps using Streamlit

---

## 👤 Author
**Vishvas Parmar**

Machine Learning & Data Science Enthusiast

---

⭐ If you like this project, consider giving it a star!

