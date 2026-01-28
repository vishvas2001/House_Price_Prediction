import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

# Load data
df = pd.read_csv("ml/data/Housing.csv")

X = df.drop("price", axis=1)
y = df["price"]

# Identify feature types
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
    ]
)

# Pipeline
pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        ("model", Ridge())
    ]
)

# Hyperparameter tuning
param_grid = {
    "model__alpha": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid.fit(X, y)

print("Best alpha:", grid.best_params_)

# Save model
joblib.dump(grid.best_estimator_, "./streamlit_app/model.pkl")
print("Model saved successfully.")
