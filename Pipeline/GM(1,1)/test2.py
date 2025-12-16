# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# COMPANY FILES
# ============================================================
COMPANY_FILES = {
    "AAPL": "D:/S-P-500-development/raw data/Apple Inc/AAPL 5y.csv",
    "MSFT": "D:/S-P-500-development/raw data/Microsoft/MSFT 5y.csv",
    "NFLX": "D:/S-P-500-development/raw data/Netflix/NFLX 5y.csv",
    "IBM": "D:/S-P-500-development/raw data/IBM/IBM 5y.csv",
    "META": "D:/S-P-500-development/raw data/Meta Platforms/META 5y.csv",
    "NVDA": "D:/S-P-500-development/raw data/Nvidia/NVDA 5y.csv",
    "ORCL": "D:/S-P-500-development/raw data/Oracle Corporation/ORCL 5y.csv",
    "TSLA": "D:/S-P-500-development/raw data/Tesla Inc/TSLA 5y.csv",
    "INTC": "D:/S-P-500-development/raw data/Intel/INTC 5y.csv",
}

FORECAST_STEPS = 5
LOOKBACK = 50
EPOCHS = 600

# ============================================================
# GM(1,1) MODEL (CLEAN DESIGN)
# ============================================================
class GreyModelGM11_NumPy:
    def __init__(self, log_file_path=None):
        if log_file_path:
            logging.basicConfig(filename=log_file_path, level=logging.INFO)
        self.a = None
        self.b = None
        self.X0 = None
        self.X0_start = None

    def fit(self, X0):
        self.X0 = np.asarray(X0)
        self.X0_start = self.X0[0]

        X1 = np.cumsum(self.X0)
        Z1 = 0.5 * (X1[1:] + X1[:-1])

        B = np.vstack((-Z1, np.ones(len(Z1)))).T
        Y = self.X0[1:].reshape(-1, 1)

        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        self.a, self.b = params.flatten()

    def _x1(self, k):
        return (self.X0_start - self.b / self.a) * np.exp(-self.a * (k - 1)) + self.b / self.a

    def forecast(self, steps):
        n = len(self.X0)
        X1_hat = np.array([self._x1(k) for k in range(1, n + steps + 1)])
        X0_hat = X1_hat[1:] - X1_hat[:-1]
        return X0_hat[:n], X0_hat[n-1:n-1+steps]


# ============================================================
# LSTM MODEL
# ============================================================
class LSTMResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, num_layers=3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_dataset(data, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        Y.append(data[i + lookback])
    return np.array(X), np.array(Y)


# ============================================================
# HYBRID GM + LSTM
# ============================================================
def hybrid_gm_lstm(prices, train_len):
    train = prices[:train_len]
    test = prices[train_len:]

    scaler_price = MinMaxScaler()
    train_scaled = scaler_price.fit_transform(train.reshape(-1, 1)).flatten()

    gm = GreyModelGM11_NumPy()
    gm.fit(train_scaled)
    gm_fit, gm_forecast = gm.forecast(FORECAST_STEPS)

    residuals = train_scaled - gm_fit
    scaler_res = MinMaxScaler()
    res_scaled = scaler_res.fit_transform(residuals.reshape(-1, 1)).flatten()

    X, Y = create_dataset(res_scaled, LOOKBACK)
    X = torch.tensor(X).float().unsqueeze(-1)
    Y = torch.tensor(Y).float().unsqueeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMResidualModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()

    X, Y = X.to(device), Y.to(device)

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        optimizer.step()

    model.eval()
    seq = res_scaled[-LOOKBACK:].copy()
    preds = []

    for _ in range(FORECAST_STEPS):
        inp = torch.tensor(seq.reshape(1, LOOKBACK, 1)).float().to(device)
        with torch.no_grad():
            p = model(inp).cpu().numpy()[0, 0]
        preds.append(p)
        seq = np.append(seq[1:], p)

    res_forecast = scaler_res.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    hybrid_scaled = gm_forecast + res_forecast

    return (
        scaler_price.inverse_transform(hybrid_scaled.reshape(-1, 1)).flatten(),
        scaler_price.inverse_transform(gm_forecast.reshape(-1, 1)).flatten(),
        test,
        train[-1]
    )


# ============================================================
# RUN ALL COMPANIES + PRINT TABLES
# ============================================================
RESULTS = {}

for ticker, path in COMPANY_FILES.items():
    print(f"\n🚀 Processing {ticker}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df[df["Date"] >= df["Date"].max() - pd.DateOffset(months=3)]

    prices = df["Close"].values
    dates = df["Date"].values

    if len(prices) <= FORECAST_STEPS + LOOKBACK:
        print("⚠️ Not enough data, skipped")
        continue

    train_len = len(prices) - FORECAST_STEPS

    P_hybrid, P_gm, P_actual, last_price = hybrid_gm_lstm(prices, train_len)

    forecast_dates = dates[train_len:]

    table = pd.DataFrame({
        "Date": pd.to_datetime(forecast_dates).strftime("%Y-%m-%d"),
        "Actual": P_actual.round(4),
        "GM(1,1)": P_gm.round(4),
        "GM + LSTM": P_hybrid.round(4)
    })

    print("\n📋 Forecast Comparison Table")
    print(table.to_string(index=False))

    RESULTS[ticker] = (dates, prices, P_hybrid, P_gm, last_price)


# ============================================================
# PLOTTING
# ============================================================
cols = 3
rows = int(np.ceil(len(RESULTS) / cols))

fig, axes = plt.subplots(rows, cols, figsize=(22, 4 * rows))
axes = axes.flatten()

for ax, (ticker, (dates, prices, hyb, gm, last_price)) in zip(axes, RESULTS.items()):
    x = np.arange(len(dates))
    labels = pd.to_datetime(dates).strftime("%m-%d")

    ax.plot(x, prices, color="black", label="History")

    start = len(dates) - len(hyb) - 1
    fx = np.arange(start, start + len(hyb) + 1)

    ax.plot(fx, np.concatenate([[last_price], gm]), ":", label="GM(1,1)")
    ax.plot(fx, np.concatenate([[last_price], hyb]), "--", label="GM + LSTM")

    ax.set_title(ticker)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.grid(True)

for i in range(len(RESULTS), len(axes)):
    fig.delaxes(axes[i])

fig.legend(["History", "GM(1,1)", "GM + LSTM"], loc="upper center", ncol=3)
fig.suptitle("GM(1,1) vs GM + LSTM Forecast (Last 3 Months)", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()