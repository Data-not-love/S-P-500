import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split 
import time 

# ============================================================
# ⚙️ CẤU HÌNH SIÊU THAM SỐ 
# ============================================================

# Cấu hình dự báo
DATA_MONTHS = 18       
TRAIN_RATIO = 0.7       # Tỷ lệ chia tập Huấn luyện/Kiểm tra

# Cấu hình làm mịn (Xử lý nhiễu)
EMA_WINDOW = 10        # Cửa sổ EMA làm mịn (10 ngày)

# Cấu hình LSTM
LOOKBACK = 75          
LSTM_EPOCHS = 600     
LR = 0.0001            
HIDDEN_SIZE = 256      
NUM_LAYERS = 3         
DROPOUT = 0.15          
BATCH_SIZE = 256        

# Cấu hình Dừng Sớm (Early Stopping)
VAL_SPLIT = 0.15        
PATIENCE = 200          

# Cấu hình GPU
USE_CUDA = True        

# ============================================================
# CÁC HÀM VÀ LỚP (Giữ nguyên)
# ============================================================

class NullLogGenerator:
    def __init__(self, log_file_path=None):
        pass
    def log_config(self):
        pass

log_generator = NullLogGenerator

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# Đảm bảo bạn cập nhật đường dẫn này nếu cần
COMPANY_FILES = {
    "AAPL": "D:/S-P-500-development/raw data/Apple Inc/AAPL 5y.csv",
    "MSFT": "D:/S-P-500-development/raw data/Microsoft/MSFT 5y.csv",
    "NFLX": "D:/S-P-500-development/raw data/Netflix/NFLX 5y.csv",
    "IBM":  "D:/S-P-500-development/raw data/IBM/IBM 5y.csv",
    "META": "D:/S-P-500-development/raw data/Meta Platforms/META 5y.csv",
    "NVDA": "D:/S-P-500-development/raw data/Nvidia/NVDA 5y.csv",
    "ORCL": "D:/S-P-500-development/raw data/Oracle Corporation/ORCL 5y.csv",
    "TSLA": "D:/S-P-500-development/raw data/Tesla Inc/TSLA 5y.csv",
    "INTC": "D:/S-P-500-development/raw data/Intel/INTC 5y.csv",
}


class GreyModelGM11_NumPy:
    def __init__(self):
        
        self.__device = torch.device('cuda') if USE_CUDA and torch.cuda.is_available() else torch.device('cpu')
        
        if self.__device.type == 'cuda':
            print(f"VGA : {torch.cuda.get_device_name(0)}, CUDA Version : {torch.version.cuda}")
        elif USE_CUDA:
            print("⚠️ CUDA is not available, using CPU")
        else:
            print("🖥️ Using CPU")

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

        try:
            params = np.linalg.inv(B.T @ B) @ B.T @ Y
            self.a = params[0, 0]
            self.b = params[1, 0]
        except np.linalg.LinAlgError:
            self.a = None
            self.b = None
            raise 

    def _x1(self, k):
        
        if self.a is None or self.b is None:
            return np.nan
            
        if self.a == 0:
            return np.nan 
        
        return (self.X0_start - self.b / self.a) * np.exp(-self.a * (k - 1)) + self.b / self.a

    def forecast(self, steps):
        if self.X0 is None:
            raise ValueError("Lỗi: GM(1,1) forecast gọi khi self.X0 chưa được gán.")
            
        n = len(self.X0) 
        
        X1_hat = np.array([self._x1(k) for k in range(1, n + steps + 1)])
        
        if np.isnan(X1_hat).any():
             nan_array = np.full(n + steps - 1, np.nan)
             return nan_array[:n], nan_array[n-1:n-1+steps]

        X0_hat = X1_hat[1:] - X1_hat[:-1]
        return X0_hat[:n], X0_hat[n-1:n-1+steps]


# Lớp LSTMResidualModel và các hàm phụ trợ giữ nguyên...
class LSTMResidualModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_dataset(data, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        Y.append(data[i + lookback])
    return np.array(X), np.array(Y)


def calculate_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    valid_indices = ~np.isnan(predicted)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    
    if len(actual) == 0:
        return np.nan, np.nan, np.nan
        
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    non_zero_indices = actual != 0
    if np.sum(non_zero_indices) == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((actual[non_zero_indices] - predicted[non_zero_indices]) / actual[non_zero_indices])) * 100
        
    return rmse, mae, mape


def hybrid_gm_lstm_forecast_actual(prices, train_len, forecast_steps, lookback=LOOKBACK, epochs=LSTM_EPOCHS, lr=LR):
    
    start_time = time.time()
    
    train = prices[:train_len]
    test = prices[train_len:]

    scaler_price = MinMaxScaler()
    train_scaled = scaler_price.fit_transform(train.reshape(-1, 1)).flatten()

    gm = GreyModelGM11_NumPy() 
    
    try:
        gm.fit(train_scaled)
    except np.linalg.LinAlgError:
        print("⚠️ GM(1,1) fit thất bại do lỗi Đại số Tuyến tính.")
        return None, None, test, train[-1], 0.0
    
    if gm.X0 is None or gm.a is None or gm.b is None or gm.a == 0:
        print("⚠️ GM(1,1) fit thất bại do tham số không hợp lệ (a=0 hoặc None).")
        return None, None, test, train[-1], 0.0
    
    gm_fit, gm_forecast = gm.forecast(forecast_steps)
    
    if np.isnan(gm_forecast).all():
        print("⚠️ GM(1,1) dự báo thất bại (NaN).")
        return None, None, test, train[-1], 0.0
        
    valid_indices = ~np.isnan(gm_fit)
    if np.sum(valid_indices) < lookback + 1:
        print(f"⚠️ Không đủ dữ liệu hợp lệ ({np.sum(valid_indices)} < {lookback+1}) sau GM fit để chạy LSTM.")
        P_gm = scaler_price.inverse_transform(gm_forecast.reshape(-1, 1)).flatten()
        return None, P_gm, test, train[-1], 0.0

    residuals = train_scaled[valid_indices] - gm_fit[valid_indices]
    
    scaler_res = MinMaxScaler()
    res_scaled = scaler_res.fit_transform(residuals.reshape(-1, 1)).flatten()

    # 3. Tạo tập dữ liệu và Huấn luyện LSTM
    X, Y = create_dataset(res_scaled, lookback)
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=VAL_SPLIT, shuffle=False
    )
    
    X_train_t = torch.tensor(X_train).float().unsqueeze(-1)
    Y_train_t = torch.tensor(Y_train).float().unsqueeze(-1)
    X_val_t = torch.tensor(X_val).float().unsqueeze(-1)
    Y_val_t = torch.tensor(Y_val).float().unsqueeze(-1)
    
    train_data = TensorDataset(X_train_t, Y_train_t)
    val_data = TensorDataset(X_val_t, Y_val_t)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMResidualModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    min_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    actual_epochs = epochs 

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_val_batch, Y_val_batch in val_loader:
                X_val_batch, Y_val_batch = X_val_batch.to(device), Y_val_batch.to(device)
                val_output = model(X_val_batch)
                val_loss_sum += loss_fn(val_output, Y_val_batch).item()

        avg_val_loss = val_loss_sum / len(val_loader)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4}/{epochs} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                actual_epochs = epoch + 1
                print(f"⚠️ Dừng sớm tại Epoch {epoch + 1}. Val Loss không cải thiện trong {PATIENCE} epochs.")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)

    end_time = time.time() 
    execution_time = end_time - start_time
    print(f"✅ Huấn luyện hoàn tất (Epochs: {actual_epochs}). Thời gian chạy: {execution_time:.2f} giây.")
    
    # 4. Dự báo Phần dư bằng LSTM
    model.eval()
    
    seq = res_scaled[-lookback:].copy() 
    preds = []

    for _ in range(forecast_steps):
        inp = torch.tensor(seq.reshape(1, lookback, 1)).float().to(device)
        with torch.no_grad():
            p = model(inp).cpu().numpy()[0, 0]
        preds.append(p)
        seq = np.append(seq[1:], p)

    # 5. Nghịch đảo chuẩn hóa và Kết hợp
    res_forecast = scaler_res.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    hybrid_scaled = gm_forecast + res_forecast 
    
    P_hybrid = scaler_price.inverse_transform(hybrid_scaled.reshape(-1, 1)).flatten()
    P_gm = scaler_price.inverse_transform(gm_forecast.reshape(-1, 1)).flatten()

    return P_hybrid, P_gm, test, train[-1], execution_time


# ============================================================
# RUN ALL COMPANIES + PRINT TABLES 
# ============================================================
RESULTS = {}
PERFORMANCE_SUMMARY = [] 
total_execution_time = 0.0

for ticker, path in COMPANY_FILES.items():
    print(f"\n{'='*5} Processing {ticker} {'='*5}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df[df["Date"] >= df["Date"].max() - pd.DateOffset(months=DATA_MONTHS)].copy() 

    df['Close_EMA'] = df['Close'].ewm(span=EMA_WINDOW, adjust=False).mean()
    
    prices = df["Close_EMA"].values 
    dates = df["Date"].values
    actual_prices_full = df["Close"].values 

    first_valid_index = np.where(~np.isnan(prices))[0][0]
    prices = prices[first_valid_index:]
    dates = dates[first_valid_index:]
    actual_prices_full = actual_prices_full[first_valid_index:]

    # 🔥 TÍNH TOÁN TRAIN_LEN VÀ FORECAST_STEPS DỰA TRÊN RATIO 🔥
    total_len = len(prices)
    train_len = int(total_len * TRAIN_RATIO)
    forecast_steps = total_len - train_len
    
    if total_len < LOOKBACK + 2 or train_len <= LOOKBACK or forecast_steps == 0:
        print(f"⚠️ Dữ liệu quá ngắn ({total_len} điểm) sau khi chia tỷ lệ ({TRAIN_RATIO*100}%), bỏ qua.")
        continue

    # Chạy mô hình lai
    P_hybrid, P_gm, P_ema_actual, last_price_ema, run_time = hybrid_gm_lstm_forecast_actual(
        prices, train_len, forecast_steps=forecast_steps
    )
    
    P_actual_test = actual_prices_full[train_len:]
    last_price_actual = actual_prices_full[train_len-1]

    total_execution_time += run_time 

    if P_hybrid is None or P_gm is None or np.isnan(P_gm).all() or np.isnan(P_hybrid).all():
        print(f"⚠️ Bỏ qua {ticker} do lỗi dự báo.")
        continue

    # Tính toán Metrics và in bảng (Giữ nguyên)
    rmse_gm, mae_gm, mape_gm = calculate_metrics(P_actual_test, P_gm)
    rmse_hyb, mae_hyb, mape_hyb = calculate_metrics(P_actual_test, P_hybrid)
    
    PERFORMANCE_SUMMARY.append({
        "Ticker": ticker,
        "GM(1,1) RMSE": rmse_gm,
        "Hybrid RMSE": rmse_hyb,
        "GM(1,1) MAE": mae_gm,
        "Hybrid MAE": mae_hyb,
        "GM(1,1) MAPE (%)": mape_gm,
        "Hybrid MAPE (%)": mape_hyb,
        "Time (s)": run_time 
    })

    forecast_dates = dates[train_len:]
    table = pd.DataFrame({
        "Date": [pd.to_datetime(d).strftime("%Y-%m-%d") for d in forecast_dates],
        "Actual": P_actual_test.round(4), 
        "GM(1,1)": P_gm.round(4),
        "GM + LSTM": P_hybrid.round(4),
    })

    print("\n📋 Bảng So sánh Dự báo (Giá trị tuyệt đối)")
    print(table.to_string(index=False))
    
    RESULTS[ticker] = (dates, actual_prices_full, prices, P_hybrid, P_gm, last_price_actual, last_price_ema)


# ============================================================
# BẢNG TỔNG KẾT HIỆU SUẤT TẤT CẢ CÔNG TY (Giữ nguyên)
# ============================================================

if PERFORMANCE_SUMMARY:
    print("\n\n" + "="*80)
    print("📊 BẢNG TỔNG KẾT CÁC CHỈ SỐ LỖI VÀ THỜI GIAN CHẠY")
    print(f" (Train Ratio = {TRAIN_RATIO*100}%, EMA Window = {EMA_WINDOW})")
    print("="*80)

    summary_df = pd.DataFrame(PERFORMANCE_SUMMARY)

    for col in summary_df.columns:
        if col != "Ticker" and col != "Time (s)":
            summary_df[col] = summary_df[col].round(4)
        elif col == "Time (s)":
            summary_df[col] = summary_df[col].round(2)


    summary_df["RMSE Cải thiện (%)"] = (
        (summary_df["GM(1,1) RMSE"] - summary_df["Hybrid RMSE"]) / summary_df["GM(1,1) RMSE"] * 100
    ).round(2)
    summary_df["Tốt hơn (Lỗi)"] = np.where(summary_df["Hybrid RMSE"] < summary_df["GM(1,1) RMSE"], "Hybrid", "GM(1,1)")

    print(summary_df.to_string(index=False))
    print(f"\n⚡ Tổng thời gian chạy cho tất cả mô hình: {total_execution_time:.2f} giây.")

# ============================================================
# PLOTTING (ĐÃ KHẮC PHỤC: CHỈ 1 CHÚ THÍCH CHUNG)
# ============================================================
if len(RESULTS) > 0:
    cols = 3
    rows = int(np.ceil(len(RESULTS) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(22, 4 * rows))
    axes = axes.flatten()
    
    # Biến để lưu trữ đối tượng đồ họa và nhãn cho chú thích chung
    legend_handles = []
    legend_labels = []

    for idx, (ax, (ticker, (dates, actual_prices_full, ema_prices, hyb, gm, last_price_actual, last_price_ema))) in enumerate(zip(axes, RESULTS.items())):

        if hyb is None or gm is None: 
            continue
            
        x = np.arange(len(dates))
        all_dates = pd.to_datetime(dates)
        
        # LOGIC TRỤC X (Giữ nguyên)
        tick_indices = []
        tick_labels = []
        
        month_year = all_dates.to_period('M') 
        odd_months = np.unique(month_year[all_dates.month % 2 != 0])
        
        for period in odd_months:
            first_occurrence_index = np.where(month_year == period)[0][0]
            tick_indices.append(first_occurrence_index)
            tick_labels.append(all_dates[first_occurrence_index].strftime("%m/%y"))

        
        current_forecast_steps = len(hyb) 
        start_index = len(dates) - current_forecast_steps 
        fx = np.arange(start_index, start_index + current_forecast_steps + 1)
        start_point_index = start_index - 1
        
        # 1. ĐƯỜNG GIÁ GỐC (THỰC TẾ) - MÀU ĐEN 
        # Chỉ thêm label nếu đây là biểu đồ đầu tiên
        p1, = ax.plot(x, actual_prices_full, color="black", linewidth=1, linestyle='-', alpha=0.5, label="Giá Gốc (Actual)" if idx == 0 else "_nolegend_")
        
        # 3. ĐƯỜNG GM(1,1) DỰ ĐOÁN (Màu Cam)
        p3, = ax.plot(fx, np.concatenate([[last_price_ema], gm]), linestyle=":", color="orange", label="GM(1,1) Forecast" if idx == 0 else "_nolegend_")
        
        # 2. Đường DỰ ĐOÁN HYBRID (Màu Xanh Dương)
        p2, = ax.plot(fx, np.concatenate([[last_price_ema], hyb]), linestyle="--", linewidth=2, color="blue", label="Hybrid (GM+LSTM) Forecast" if idx == 0 else "_nolegend_")
        
        # Hoàn thiện đường giá gốc trong vùng test
        ax.plot(x[start_index:], actual_prices_full[start_index:], color='black', linewidth=1.5, label='_nolegend_') 

        ax.set_title(f"{ticker} (RMSE: {calculate_metrics(P_actual_test, hyb)[0]:.2f}, MAE: {calculate_metrics(P_actual_test, hyb)[1]:.2f})")
        
        if len(tick_indices) > 0:
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 🔥 THU THẬP THÔNG TIN CHÚ THÍCH TỪ BIỂU ĐỒ ĐẦU TIÊN
        if idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            
            # Loại bỏ các handle trùng lặp nếu có (giữ nguyên thứ tự ban đầu)
            unique_handles = []
            unique_labels = []
            for h, l in zip(legend_handles, legend_labels):
                if l not in unique_labels and l != "_nolegend_":
                    unique_labels.append(l)
                    unique_handles.append(h)
            legend_handles = unique_handles
            legend_labels = unique_labels

    for i in range(len(RESULTS), len(axes)):
        fig.delaxes(axes[i])
        
    fig.suptitle(f"So sánh Dự báo Giá Cổ Phiếu GM(1,1) vs GM + LSTM", fontsize=16)
    
    # Điều chỉnh layout để tạo không gian cho chú thích chung
    plt.tight_layout(rect=(0, 0, 1, 0.94)) 
    plt.subplots_adjust(
        wspace=0.2, 
        hspace=0.50,
    )

    # 🔥 TẠO CHÚ THÍCH CHUNG CHO TOÀN BỘ HÌNH ẢNH (OUTSIDE SUBPLOTS)
    if len(legend_handles) > 0:
        fig.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(0.5, 0.98), ncol=len(legend_labels), fontsize=10)

    plt.show()