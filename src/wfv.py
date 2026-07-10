import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv("Nifty_50_Historical_Data_ISO.csv")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["Price"] = df["Price"].astype(str).str.replace(",", "").astype(float)

df = df.sort_values("Date").reset_index(drop=True)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

def mape(a,b):
    return np.mean(np.abs((a-b)/a))*100

def directional_accuracy(actual, predicted, previous_price):

    actual_direction = np.sign(actual - previous_price)
    predicted_direction = np.sign(predicted - previous_price)

    return np.mean(actual_direction == predicted_direction) * 100


# ------------------------------------------------------------
# LSTM forecast function
# ------------------------------------------------------------

def lstm_forecast(train_prices, steps, window=10):

    # Convert price to returns
    returns = np.diff(train_prices) / train_prices[:-1]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(returns.reshape(-1,1))

    X=[]
    y=[]

    for i in range(window,len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    X=np.array(X)
    y=np.array(y)

    model = Sequential([
        Input(shape=(window,1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam",loss="mse")

    model.fit(X,y,epochs=10,verbose=0)

    history=list(scaled)

    preds=[]

    for _ in range(steps):

        seq=np.array(history[-window:])
        seq=seq.reshape(1,window,1)

        pred=model.predict(seq,verbose=0)[0][0]

        preds.append(pred)

        history.append([pred])

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

    # Convert returns back to price
    last_price=train_prices[-1]

    prices=[]
    for r in preds:
        last_price = last_price*(1+r)
        prices.append(last_price)

    return np.array(prices)

# ------------------------------------------------------------
# Walk-forward validation
# ------------------------------------------------------------

all_dates=[]
all_actual=[]
all_ar=[]
all_hybrid=[]
all_lstm=[]

yearly_metrics=[]


for base_year in range(2007,2022):

    past=df[df["Date"].dt.year<=base_year]["Price"].values
    future=df[df["Date"].dt.year==base_year+1]

    if len(future)<30:
        continue

    actual=future["Price"].values

    # ----------------------------
    # AR model
    # ----------------------------

    ar=AutoReg(past,lags=5).fit()


    # ----------------------------
    # Train RF + GB on residuals
    # ----------------------------

    X=[]
    y=[]

    for i in range(10,len(past)):

        X.append([
            past[i-1],
            past[i-2],
            past[i-3],
            past[i-5],
            np.mean(past[i-5:i]),
            np.mean(past[i-10:i]),
            np.std(past[i-10:i])
        ])

        y.append(past[i]-ar.predict(start=i,end=i)[0])


    rf=RandomForestRegressor(n_estimators=80,max_depth=6,random_state=42)
    gb=GradientBoostingRegressor(n_estimators=120,learning_rate=0.05,max_depth=3)

    rf.fit(X,y)
    gb.fit(X,y)


    # ----------------------------
    # LSTM predictions
    # ----------------------------

    lstm_preds=lstm_forecast(past,len(actual))


    # ----------------------------
    # Recursive forecasting
    # ----------------------------

    history=list(past)

    ar_preds=[]
    hybrid_preds=[]
    lstm_only_preds=[]


    for i in range(len(actual)):

        n=len(history)

        ar_p=ar.predict(start=n,end=n)[0]

        feats=[
            history[-1],
            history[-2],
            history[-3],
            history[-5],
            np.mean(history[-5:]),
            np.mean(history[-10:]),
            np.std(history[-10:])
        ]

        rf_res=rf.predict([feats])[0]
        gb_res=gb.predict([feats])[0]

        hybrid_p = ar_p + (rf_res + gb_res)/2

        lstm_p = lstm_preds[i]


        ar_preds.append(ar_p)
        hybrid_preds.append(hybrid_p)
        lstm_only_preds.append(lstm_p)

        history.append(hybrid_p)


    # ----------------------------
    # Metrics
    # ----------------------------

    rmse_ar = rmse(actual,ar_preds)
    rmse_hybrid = rmse(actual,hybrid_preds)
    rmse_lstm = rmse(actual,lstm_only_preds)
    previous_prices = np.concatenate(([past[-1]], actual[:-1]))

    da_ar = directional_accuracy(
        actual,
        np.array(ar_preds),
        previous_prices
    )

    da_hybrid = directional_accuracy(
        actual,
        np.array(hybrid_preds),
        previous_prices
    )

    da_lstm = directional_accuracy(
        actual,
        np.array(lstm_only_preds),
        previous_prices
    )

    print(
    f"{base_year+1} | "
    f"AR RMSE:{rmse_ar:.2f} DA:{da_ar:.2f}% | "
    f"Hybrid RMSE:{rmse_hybrid:.2f} DA:{da_hybrid:.2f}% | "
    f"LSTM RMSE:{rmse_lstm:.2f} DA:{da_lstm:.2f}%"
)

    yearly_metrics.append({

    "Year":base_year+1,

    "AR_RMSE":rmse_ar,
    "Hybrid_RMSE":rmse_hybrid,
    "LSTM_RMSE":rmse_lstm,

    "AR_MAPE":mape(actual,ar_preds),
    "Hybrid_MAPE":mape(actual,hybrid_preds),
    "LSTM_MAPE":mape(actual,lstm_only_preds),

    "AR_Directional_Accuracy":da_ar,
    "Hybrid_Directional_Accuracy":da_hybrid,
    "LSTM_Directional_Accuracy":da_lstm,

})


    all_dates.extend(future["Date"].values)

    all_actual.extend(actual)
    all_ar.extend(ar_preds)
    all_hybrid.extend(hybrid_preds)
    all_lstm.extend(lstm_only_preds)


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results=pd.DataFrame({

"Date":all_dates,
"Actual":all_actual,
"AR":all_ar,
"Hybrid":all_hybrid,
"LSTM":all_lstm,

})

results.to_csv("walk_forward_predictions.csv",index=False)

pd.DataFrame(yearly_metrics).to_excel("walk_forward_metrics.xlsx",index=False)


# ------------------------------------------------------------
# Plot comparison
# ------------------------------------------------------------

plt.figure(figsize=(14,6))

plt.plot(results["Date"],results["Actual"],label="Actual",linewidth=2)

plt.plot(results["Date"],results["AR"],label="AR(5)",linestyle=":")
plt.plot(results["Date"],results["Hybrid"],label="Hybrid (AR+RF+GB)",linestyle="--")

plt.title("Walk-Forward Validation Comparison")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()