import pandas as pd
import joblib
from sklearn.metrics import root_mean_squared_error, r2_score

df = pd.read_csv("ml/data/Housing.csv")

X = df.drop("price", axis=1)
y = df["price"]

model = joblib.load("app/backend/model/model.pkl")

preds = model.predict(X)

print("RMSE:", root_mean_squared_error(y, preds))
print("R2:", r2_score(y, preds))
