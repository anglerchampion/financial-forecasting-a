import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load & clean
df = pd.read_csv("Nifty_50_Historical_Data_ISO.csv")
df['Date']  = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Price'] = df['Price'].astype(str).str.replace(",", "").astype(float)
df = df.sort_values("Date").reset_index(drop=True)

all_dates, all_actual, all_ar, all_hybrid = [], [], [], []
yearly_metrics = []

for base_year in range(2007, 2022):

    past   = df[df['Date'].dt.year <= base_year]['Price'].values
    future = df[df['Date'].dt.year == base_year + 1]

    if len(future) < 30:
        continue

    actual = future['Price'].values

    # AR(5)
    ar = AutoReg(past, lags=5).fit()

    # Train ML models on AR residuals
    X, y = [], []

    for i in range(10, len(past)):

        features = [
            past[i-1],
            past[i-2],
            past[i-3],
            past[i-5],
            np.mean(past[i-5:i]),
            np.mean(past[i-10:i]),
            np.std(past[i-10:i])
        ]

        residual = past[i] - ar.predict(start=i, end=i)[0]

        X.append(features)
        y.append(residual)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=80,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X, y)

    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    gb.fit(X, y)

    # Recursive forecast
    history = list(past)
    ar_preds = []
    hybrid_preds = []

    for _ in range(len(actual)):

        n = len(history)

        # AR prediction
        ar_p = ar.predict(start=n, end=n)[0]

        feats = [
            history[-1],
            history[-2],
            history[-3],
            history[-5],
            np.mean(history[-5:]),
            np.mean(history[-10:]),
            np.std(history[-10:])
        ]

        # ML residual corrections
        rf_res = rf.predict([feats])[0]
        gb_res = gb.predict([feats])[0]

        # Hybrid correction
        hybrid_p = ar_p + (rf_res + gb_res) / 2

        ar_preds.append(ar_p)
        hybrid_preds.append(hybrid_p)

        history.append(hybrid_p)

    ar_preds = np.array(ar_preds)
    hybrid_preds = np.array(hybrid_preds)

    # Metrics
    def mape(a, b):
        return np.mean(np.abs((a - b) / a)) * 100

    def dacc(a, b):
        return np.mean(np.sign(np.diff(a)) == np.sign(np.diff(b))) * 100

    rmse_ar = np.sqrt(mean_squared_error(actual, ar_preds))
    rmse_hy = np.sqrt(mean_squared_error(actual, hybrid_preds))

    print(
        f"{base_year+1} | "
        f"AR RMSE: {rmse_ar:.2f}  Hybrid RMSE: {rmse_hy:.2f} | "
        f"AR MAPE: {mape(actual, ar_preds):.2f}%  Hybrid MAPE: {mape(actual, hybrid_preds):.2f}% | "
        f"AR Dir: {dacc(actual, ar_preds):.2f}%  Hybrid Dir: {dacc(actual, hybrid_preds):.2f}%"
    )

    yearly_metrics.append({
        "Year": base_year + 1,
        "AR RMSE": rmse_ar,
        "Hybrid RMSE": rmse_hy,
        "AR MAPE": mape(actual, ar_preds),
        "Hybrid MAPE": mape(actual, hybrid_preds),
        "AR Directional Accuracy": dacc(actual, ar_preds),
        "Hybrid Directional Accuracy": dacc(actual, hybrid_preds)
    })

    all_dates.extend(future['Date'].values)
    all_actual.extend(actual)
    all_ar.extend(ar_preds)
    all_hybrid.extend(hybrid_preds)

# Save results
results = pd.DataFrame({
    "Date": all_dates,
    "Actual_Price": all_actual,
    "AR_Prediction": all_ar,
    "Hybrid_Prediction": all_hybrid
})

results["Absolute_Error"] = abs(results["Actual_Price"] - results["Hybrid_Prediction"])
results["MAPE"] = results["Absolute_Error"] / results["Actual_Price"] * 100

results["Risk_Regime"] = np.where(
    results["MAPE"] > 15,
    "High Risk",
    np.where(results["MAPE"] > 7, "Medium Risk", "Low Risk")
)

results.to_csv("walk_forward_results.csv", index=False)
pd.DataFrame(yearly_metrics).to_excel("walk_forward_metrics.xlsx", index=False)

# Plot
plt.figure(figsize=(14,6))

plt.plot(results["Date"], results["Actual_Price"], label="Actual", linewidth=2)
plt.plot(results["Date"], results["AR_Prediction"], label="AR(5)", linestyle=":")
plt.plot(results["Date"], results["Hybrid_Prediction"], label="Hybrid (AR+RF+GB)", linestyle="--")

plt.title("Walk-Forward Validation: AR vs Hybrid Forecast")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("walk_forward_ar_vs_hybrid.png", dpi=300)

plt.show()