import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ===============================
# Data Cleaning and Loading
# ===============================

df = pd.read_csv("Nifty_50_Historical_Data_ISO.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date").reset_index(drop=True)

df['Price'] = df['Price'].astype(str).str.replace(",", "")
df['Price'] = df['Price'].astype(float)

# ===============================
# Walk Forward
# ===============================

all_dates, all_actuals = [], []
all_hybrid_preds, all_ar_preds = [], []

for year in range(2007, 2022):

    train = df[df['Date'].dt.year <= year]
    test  = df[df['Date'].dt.year == year+1]

    if len(test) < 30:
        continue

    train_y = train['Price'].values
    test_y  = test['Price'].values
    test_dates = test['Date'].values

    # ---------------------------
    # AR(5)
    # ---------------------------
    ar = AutoReg(train_y, lags=5).fit()

    # ---------------------------
    # Residual dataset for RF
    # ---------------------------
    X_train, y_train_res = [], []

    for i in range(10, len(train_y)):
        ar_pred = ar.predict(start=i, end=i)[0]
        res = train_y[i] - ar_pred

        X_train.append([
            train_y[i-1], train_y[i-2], train_y[i-3], train_y[i-5],
            np.mean(train_y[i-5:i]),
            np.mean(train_y[i-10:i]),
            np.std(train_y[i-10:i])
        ])
        y_train_res.append(res)

    X_train = np.array(X_train)
    y_train_res = np.array(y_train_res)

    rf = RandomForestRegressor(
        n_estimators=80,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_res)

    # ---------------------------
    # Recursive Forecasts
    # ---------------------------
    history = list(train_y)

    hybrid_preds = []
    ar_preds = []

    for i in range(len(test_y)):
        ar_p = ar.predict(start=len(history), end=len(history))[0]
        ar_preds.append(ar_p)

        row = [
            history[-1], history[-2], history[-3], history[-5],
            np.mean(history[-5:]),
            np.mean(history[-10:]),
            np.std(history[-10:])
        ]

        rf_corr = rf.predict([row])[0]
        hybrid_p = ar_p + rf_corr

        hybrid_preds.append(hybrid_p)
        history.append(hybrid_p)

    # ---------------------------
    # Metrics
    # ---------------------------
    rmse_ar = np.sqrt(mean_squared_error(test_y, ar_preds))
    rmse_hybrid = np.sqrt(mean_squared_error(test_y, hybrid_preds))

    mape_ar = np.mean(np.abs((test_y - ar_preds) / test_y)) * 100
    mape_hybrid = np.mean(np.abs((test_y - hybrid_preds) / test_y)) * 100

    directional_accuracy_ar = np.mean(
        np.sign(test_y[1:] - test_y[:-1]) ==
        np.sign(np.array(ar_preds[1:]) - np.array(ar_preds[:-1]))
    ) * 100

    directional_accuracy_hybrid = np.mean(
        np.sign(test_y[1:] - test_y[:-1]) ==
        np.sign(np.array(hybrid_preds[1:]) - np.array(hybrid_preds[:-1]))
    ) * 100

    wf_metrics =[]

    print(
        f"{year+1} → "
        f"AR RMSE: {rmse_ar:.2f} | Hybrid RMSE: {rmse_hybrid:.2f} | "
        f"AR MAPE: {mape_ar:.2f}% | Hybrid MAPE: {mape_hybrid:.2f}% | "
        f"AR DirAcc: {directional_accuracy_ar:.2f}% | "
        f"Hybrid DirAcc: {directional_accuracy_hybrid:.2f}%"
    )

    wf_metrics.append({
        "Year": year + 1,
        "AR_RMSE": rmse_ar,
        "Hybrid_RMSE": rmse_hybrid,
        "AR_MAPE": mape_ar,
        "Hybrid_MAPE": mape_hybrid,
        "AR_Directional_Accuracy": directional_accuracy_ar,
        "Hybrid_Directional_Accuracy": directional_accuracy_hybrid
    })

    # store predictions
    all_dates.extend(test_dates)
    all_actuals.extend(test_y)
    all_hybrid_preds.extend(hybrid_preds)
    all_ar_preds.extend(ar_preds)

# ===============================
# Save results
# ===============================

wf_df = pd.DataFrame({
    "Date": all_dates,
    "Actual_Price": all_actuals,
    "AR_Prediction": all_ar_preds,
    "Hybrid_Prediction": all_hybrid_preds
})

wf_df["Absolute_Error"] = abs(
    wf_df["Actual_Price"] - wf_df["Hybrid_Prediction"]
)

wf_df["MAPE"] = wf_df["Absolute_Error"] / wf_df["Actual_Price"] * 100

wf_df["Risk_Regime"] = np.where(
    wf_df["MAPE"] > 15, "High Risk",
    np.where(wf_df["MAPE"] > 7, "Medium Risk", "Low Risk")
)

wf_df.to_csv("walk_forward_results.csv", index=False)

metrics_df = pd.DataFrame(wf_metrics)
metrics_df.to_excel("walk_forward_metrics.xlsx", index=False)

# ===============================
# Walk-forward visualization
# ===============================

plt.figure(figsize=(14,6))

plt.plot(wf_df["Date"], wf_df["Actual_Price"], label="Actual", linewidth=2)
plt.plot(wf_df["Date"], wf_df["AR_Prediction"], linestyle=":", label="AR(5)")
plt.plot(wf_df["Date"], wf_df["Hybrid_Prediction"], linestyle="--", label="Hybrid")

plt.title("Walk-Forward Validation: AR vs Hybrid Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("walk_forward_ar_vs_hybrid.png", dpi=300)
plt.show()
