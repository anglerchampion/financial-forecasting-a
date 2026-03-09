import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Metrics 

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


# Data 

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst = True, errors = 'coerce')
    df = df.sort_values('Date').set_index('Date')
    df = df.ffill()
    if 'Close' in df.columns:
        df.rename(columns={'Close': 'Price'}, inplace=True)
    return df[['Price']]

def split_data(df):
    train = df[:'2020-12-31']
    test = df['2021-01-01':'2022-12-31']
    print(f"Train: {len(train)} | Test: {len(test)}")
    return train, test

# Models training

def actual_returns(series):
    return series.pct_change().shift(-1)

def ar5_returns(train, test, lags=5):
    r = actual_returns(train['Price']).dropna()
    X, y = [], []
    for i in range(lags, len(r)):
        X.append(r.iloc[i-lags:i].values)
        y.append(r.iloc[i])
    X, y = np.array(X), np.array(y)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    history = list(r.iloc[-lags:].values)
    preds = []
    test_r = actual_returns(test['Price'])
    for i in range(len(test_r)):
        pred = np.dot(coeffs, history[-lags:])
        preds.append(pred)
        if i < len(test_r)-1:
            history.append(test_r.iloc[i])
    return pd.Series(preds, index=test_r.index)

def random_walk_returns(train, test):
    train_r = train['Price'].pct_change().dropna()
    last_r = train_r.iloc[-1]

    test_r = test['Price'].pct_change().shift(-1)
    preds = []
    for i in range(len(test_r)):
        preds.append(last_r if i == 0 else test_r.iloc[i-1])
    return pd.Series(preds, index=test_r.index)


def moving_average_returns(train, test, window=20):
    r = actual_returns(train['Price']).dropna().tolist()
    preds = []
    test_r = actual_returns(test['Price'])
    for i in range(len(test_r)):
        preds.append(np.mean(r[-window:]))
        r.append(test_r.iloc[i])
    return pd.Series(preds, index=test_r.index)

# Features for Machine Learning models

def create_features(data):
    df = pd.DataFrame(index=data.index)
    for lag in [1,2,3,5,10,20]:
        df[f'lag_p_{lag}'] = data['Price'].shift(lag)
        df[f'lag_r_{lag}'] = data['Price'].pct_change(lag)
    for w in [5,10,20,50]:
        df[f'ma_{w}'] = data['Price'].rolling(w).mean()
        df[f'std_{w}'] = data['Price'].rolling(w).std()
    df['target'] = data['Price'].pct_change().shift(-1)
    return df.dropna()

def split_feats(train, test):
    full = pd.concat([train, test])
    feats = create_features(full)
    train_f = feats.loc[feats.index <= train.index[-1]]
    test_f = feats.loc[feats.index.intersection(test.index)]
    Xtr = train_f.drop('target', axis=1)
    ytr = train_f['target']
    Xte = test_f.drop('target', axis=1)
    yte = test_f['target']
    return Xtr, ytr, Xte, yte

def rf_returns(train, test):
    Xtr, ytr, Xte, yte = split_feats(train, test)
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    rf.fit(Xtr, ytr)
    return pd.Series(rf.predict(Xte), index=yte.index)

def gb_returns(train, test):
    Xtr, ytr, Xte, yte = split_feats(train, test)
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    gb.fit(Xtr, ytr)
    return pd.Series(gb.predict(Xte), index=yte.index)

def hybrid_returns(train, test):
    ar_p = ar5_returns(train, test)
    rf_p = rf_returns(train, test)
    gb_p = gb_returns(train, test)

    aligned = pd.concat([ar_p, rf_p, gb_p], axis=1).dropna()
    aligned.columns = ["AR", "RF", "GB"]

    hybrid = aligned["AR"] + (aligned["RF"] + aligned["GB"]) / 2
    return hybrid


# Main 

def main():
    df = load_data(r"C:\8th Sem Project\Nifty_50_Historical_Data_ISO.csv")
    train, test = split_data(df)

    y_true = actual_returns(test['Price'])

    preds = {
    "AR(5)": ar5_returns(train, test),
    "Random Walk": random_walk_returns(train, test),
    "Moving Average": moving_average_returns(train, test),
    "Random Forest": rf_returns(train, test),
    "Gradient Boosting": gb_returns(train, test),
    "Hybrid (AR+RF+GB)": hybrid_returns(train, test)
}


    # Align & score

    rows = []
    for name, p in preds.items():
        aligned = pd.concat([y_true, p], axis=1).dropna()
        y, yhat = aligned.iloc[:,0], aligned.iloc[:,1]

        rows.append({
            "Model": name,
            "RMSE": rmse(y, yhat),
            "MAE": mae(y, yhat),
            "R2": r2_score(y, yhat)
        })


    fig, axes = plt.subplots(3, 2, figsize=(16,12))
    axes = axes.flatten()

    model_names = list(preds.keys())

    for i, name in enumerate(model_names):
        aligned = pd.concat([y_true, preds[name]], axis=1).dropna()
        
        axes[i].plot(aligned.index, aligned.iloc[:,0], label="Actual", linewidth=2)
        axes[i].plot(aligned.index, aligned.iloc[:,1], linestyle="--", label=name)
        
        axes[i].set_title(f"Actual vs {name}")
        axes[i].grid(alpha=0.3)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("model_prediction_subplots.png", dpi=300)
    plt.show()
    
    plt.figure(figsize=(16,8))

    plt.plot(y_true.index, y_true, label="Actual Returns", linewidth=2)

    for name, p in preds.items():
        plt.plot(p.index, p, linestyle="--", label=name)

    plt.title("All Models vs Actual Returns (2021–2022)")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("all_models_vs_actual.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
