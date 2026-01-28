import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ------------------ Paths ------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR.parent / "ml" / "data" / "Housing.csv"

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ------------------ HEADER ------------------
st.title("🏠 House Price Prediction App")
st.caption("Predict house prices using Machine Learning")
st.markdown("---")

# ------------------ MODEL PERFORMANCE ------------------
X = df.drop("price", axis=1)
y = df["price"]

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

col_m1, col_m2 = st.columns(2)
col_m1.metric("📈 R² Score", f"{r2:.3f}")
col_m2.metric("📉 RMSE", f"{rmse:,.0f}")

st.markdown("---")

# ------------------ PRICE DISTRIBUTION ------------------
st.subheader("📊 Price Distribution (Dataset)")

fig, ax = plt.subplots()
ax.hist(df["price"], bins=30)
ax.set_xlabel("House Price")
ax.set_ylabel("Frequency")

st.pyplot(fig)

st.markdown("---")

# ------------------ INPUT SECTION ------------------
st.subheader("🏡 Enter House Details")

col1, col2 = st.columns(2)

with col1:
    area = st.slider("Area (sq ft)", 500, 10000, 3000, step=100)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    stories = st.slider("Stories", 1, 5, 2)

with col2:
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    parking = st.slider("Parking Spaces", 0, 5, 1)

st.markdown("### Amenities")

col3, col4 = st.columns(2)

with col3:
    mainroad = st.selectbox("Main Road", ["yes", "no"])
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])

with col4:
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

# ------------------ PREDICTION ------------------
st.markdown("---")

if st.button("💰 Predict House Price", use_container_width=True):

    input_df = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }])

    prediction = model.predict(input_df)[0]

    st.success("### 🏷️ Estimated House Price")
    st.metric("Predicted Price (₹)", f"{prediction:,.0f}")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-Learn")
