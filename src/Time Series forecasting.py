import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

#Prepping the Data 

df = pd.read_csv("Nifty 50 Historical Data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date").reset_index(drop=True)

df['Price'] = df['Price'].astype(str).str.replace(",", "")
df['Price'] = df['Price'].astype(float)

train_df = df[df['Date'] < '2023-07-27'].copy()
train_y = train_df['Price'].values
print(f"Training until: {train_df['Date'].max().date()}")

#Feature Engineering 

def make_features(series, i):
    return [
        series[i-1], series[i-2], series[i-3], series[i-5],
        np.mean(series[i-5:i]), np.mean(series[i-10:i]), np.std(series[i-10:i])
    ]

#AR(5) Training

ar_model = AutoReg(train_y, lags=5).fit()

#Residual Dataset

X_train, y_train_res = [], []

for i in range(10, len(train_y)):
    ar_pred = ar_model.predict(start=i, end=i)[0]
    res = train_y[i] - ar_pred

    X_train.append(make_features(train_y, i))
    y_train_res.append(res)

X_train = np.array(X_train)
y_train_res = np.array(y_train_res)

# Scale residuals
scaler = StandardScaler()
y_train_res = scaler.fit_transform(y_train_res.reshape(-1,1)).ravel()

#Random Forest training 

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train_res)

#Gradient Boosting training
gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train_res)

# AR Predictor

def ar_recursive(model, history):
    lags = int(max(model.model._lags))
    coefs = model.params[1:]
    intercept = model.params[0]
    x = history[-lags:][::-1]
    return intercept + np.dot(coefs, x)

# Actual forecast 

forecast_history = list(train_y)
predictions = []

print("\n5-Day Hybrid Forecast")

for d in range(5):
    ar_p = ar_recursive(ar_model, forecast_history)

    row = [
        forecast_history[-1], forecast_history[-2], forecast_history[-3], forecast_history[-5],
        np.mean(forecast_history[-5:]),
        np.mean(forecast_history[-10:]),
        np.std(forecast_history[-10:])
    ]

    rf_res_scaled = rf.predict([row])[0]
    gb_res_scaled = gb.predict([row])[0]

    rf_res = scaler.inverse_transform([[rf_res_scaled]])[0][0]
    gb_res = scaler.inverse_transform([[gb_res_scaled]])[0][0]

    final_p = ar_p + (rf_res + gb_res) / 2

    predictions.append(final_p)       
    forecast_history.append(final_p)   

    print(f"Day {d+1}: {final_p:.2f}")

# Results 

dates = pd.bdate_range(start="2023-07-28", periods=5).strftime("%Y-%m-%d").tolist()

results = pd.DataFrame({
    "Date": dates,
    "Predicted_Price": predictions
})

# Save to Excel

dates = pd.bdate_range(start="2023-07-28", periods=5).strftime("%Y-%m-%d").tolist()

results = pd.DataFrame({
    "Date": dates,
    "Predicted_Price": predictions
})

results.to_excel("nifty_2023_forecast.xlsx", index=False)
print("\nForecast saved to nifty_2023_forecast.xlsx")
