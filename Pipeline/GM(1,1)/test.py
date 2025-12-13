import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# =========================================================================
# === KHAI BÁO DỮ LIỆU GIẢ ĐỊNH  ===
# =========================================================================

file_path = 'D:\\S-P-500-development\\raw data\\3M\\MMM 5y.csv' 
try:
    # 1. Đọc file CSV thực tế
    df = pd.read_csv(file_path)
    
    # 2. Xử lý Ngày tháng và Sắp xếp
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=True)

    # 3. TÍNH TOÁN: LẤY 3 THÁNG CUỐI CÙNG
    latest_date = df['Date'].max()
    # Tính ngày bắt đầu (3 tháng trước ngày cuối cùng)
    start_date = latest_date - pd.DateOffset(months=3)
    
    # Lọc DataFrame
    df_recent = df[df['Date'] >= start_date].copy()
    
    # Gán kết quả vào các biến
    data_X0_prices_full = df_recent['Close'].values
    dates_full = df_recent['Date'].values
    print(f"ĐÃ TẢI THÀNH CÔNG {len(data_X0_prices_full)} ngày dữ liệu (3 tháng cuối cùng) từ file.")

except FileNotFoundError:
   print(f"LỖI: Không tìm thấy file tại đường dẫn: {file_path}")
   print("Sử dụng dữ liệu mô phỏng để tiếp tục phân tích.")
   

forecast_steps = 5 # 5 ngày dự báo/kiểm tra (Giá trị cố định)
total_data_length = len(data_X0_prices_full)

if total_data_length <= forecast_steps + 1: # Cần ít nhất 6 ngày (1 train + 5 test)
    raise ValueError(f"Lỗi: Dữ liệu hiện có ({total_data_length} ngày) không đủ để tách 5 ngày kiểm tra.")

# train_len tự động scale bằng tổng số ngày trừ đi 5 ngày kiểm tra
train_len = total_data_length - forecast_steps
print(f"Train Length (train_len) tự động điều chỉnh = {train_len} ngày.")


# =========================================================================
# === PHẦN 1: ĐỊNH NGHĨA MÔ HÌNH VÀ HÀM (Cập nhật Kiến trúc LSTM) ===
# =========================================================================

# --- Định nghĩa Lớp GM(1,1) (NumPy) ---
class GreyModelGM11_NumPy:
    # (Giữ nguyên class GM(1,1) như trước)
    def __init__(self):
        self.a = None
        self.b = None
        self.X0_start = None

    def fit(self, X0):
        self.X0 = np.array(X0)
        self.X0_start = self.X0[0]
        n = len(self.X0)
        X1 = np.cumsum(self.X0)
        Z1 = 0.5 * (X1[1:] + X1[:-1])
        Y = self.X0[1:].reshape(-1, 1) 
        B = np.vstack([-Z1, np.ones(n - 1)]).T
        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        self.a = params[0, 0]
        self.b = params[1, 0]
        
    def _predict_X1(self, k):
        if self.a == 0:
            return self.X0_start + self.b * (k - 1)
        term1 = self.X0_start - self.b / self.a
        return term1 * np.exp(-self.a * (k - 1)) + self.b / self.a

    def get_fitted_and_forecast(self, steps):
        n = len(self.X0)
        total_k = n + steps 
        X1_pred_all = np.array([self._predict_X1(k) for k in range(1, total_k + 1)])
        X0_pred_all = X1_pred_all[1:] - X1_pred_all[:-1]
        
        # Trả về cả fitted values (từ index 0 đến n-1) và forecast (từ index n-1)
        fitted = X0_pred_all[:n]
        forecast = X0_pred_all[n-1:n-1+steps]
        return fitted, forecast


# --- Định nghĩa Mô hình LSTM (PyTorch) - Đã Cập nhật Hidden Size và Dropout ---
class LSTMResidualModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.2):
        super(LSTMResidualModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x) 
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# --- Hàm tạo Dữ liệu chuỗi thời gian cho LSTM (Giữ nguyên) ---
def create_dataset(data, lookback=1):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), 0])
        Y.append(data[i + lookback, 0])
    return np.array(X), np.array(Y)

# --- Hàm Tính toán Độ lỗi (Giữ nguyên) ---
def calculate_metrics(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-10, y_true)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, mae, mape

# --- Hàm Thực thi Mô hình Lai - ĐÃ VIẾT LẠI LOGIC CHUẨN HÓA GIÁ ---
def hybrid_gm_lstm_forecast_actual(data_X0_prices_full, train_len, forecast_steps=5, lookback=30, epochs=500):
    
    data_X0_prices_train = data_X0_prices_full[:train_len]
    P_actual_test = data_X0_prices_full[train_len:] 
    
    # 1. Chuẩn hóa Giá đóng cửa (0-1)
    scaler_price = MinMaxScaler(feature_range=(0, 1))
    # reshape(-1, 1) là bắt buộc cho MinMaxScaler
    P_train_scaled = scaler_price.fit_transform(data_X0_prices_train.reshape(-1, 1)).flatten()
    
    # 2. Huấn luyện GM(1,1) trên GIÁ đã chuẩn hóa
    gm11_model = GreyModelGM11_NumPy()
    gm11_model.fit(P_train_scaled)
    
    # Lấy fitted values và dự báo GM(1,1) trên dữ liệu đã chuẩn hóa
    P_fitted_gm_scaled, P_forecast_gm_scaled = gm11_model.get_fitted_and_forecast(forecast_steps)
    
    # 3. Tính Residuals (Dư thừa) trên GIÁ đã chuẩn hóa
    # Lưu ý: P_fitted_gm_scaled có độ dài n (train_len)
    residuals_scaled = P_train_scaled - P_fitted_gm_scaled
    
    # 4. Huấn luyện LSTM trên Residuals
    scaler_res = MinMaxScaler(feature_range=(0, 1)) # Scaler mới cho Residuals
    residuals_scaled_for_lstm = scaler_res.fit_transform(residuals_scaled.reshape(-1, 1))
    
    # Chuẩn bị dữ liệu cho LSTM
    X_res, Y_res = create_dataset(residuals_scaled_for_lstm, lookback)
    
    X_res_torch = torch.from_numpy(X_res).float().unsqueeze(-1)
    Y_res_torch = torch.from_numpy(Y_res).float().unsqueeze(-1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tinh chỉnh tham số LSTM: hidden_size=64, num_layers=2, dropout=0.2
    lstm_model = LSTMResidualModel(1, 128, 1, 3, dropout_rate=0.2).to(device) 
    criterion = nn.MSELoss()
    # Tinh chỉnh learning rate
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0005) 
    
    X_res_torch = X_res_torch.to(device)
    Y_res_torch = Y_res_torch.to(device)
    
    for epoch in range(epochs):
        lstm_model.train()
        outputs = lstm_model(X_res_torch)
        loss = criterion(outputs, Y_res_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    lstm_model.eval()
    
    # 5. Dự báo Residuals LSTM
    current_input = residuals_scaled_for_lstm[-lookback:].copy()
    lstm_residual_forecast_scaled = []
    
    for _ in range(forecast_steps):
        input_tensor = torch.from_numpy(current_input.reshape(1, lookback, 1)).float().to(device)
        with torch.no_grad():
            predicted_scaled = lstm_model(input_tensor).detach().cpu().numpy().flatten()[0] 
        
        lstm_residual_forecast_scaled.append(predicted_scaled)
        current_input = np.append(current_input[1:], predicted_scaled)[-lookback:]
    
    # Đảo ngược chuẩn hóa Residuals
    residual_forecast_orig_scaled = scaler_res.inverse_transform(np.array(lstm_residual_forecast_scaled).reshape(-1, 1)).flatten()
    
    # 6. KẾT HỢP VÀ ĐẢO NGƯỢC CHUẨN HÓA CUỐI CÙNG
    
    # Kết hợp GM(scaled) + Residual LSTM(scaled)
    P_hybrid_forecast_scaled = P_forecast_gm_scaled + residual_forecast_orig_scaled 
    
    # Đảo ngược chuẩn hóa GM(1,1) và Hybrid về giá USD
    P_gm_forecast_usd = scaler_price.inverse_transform(P_forecast_gm_scaled.reshape(-1, 1)).flatten()
    P_hybrid_forecast_usd = scaler_price.inverse_transform(P_hybrid_forecast_scaled.reshape(-1, 1)).flatten()
        
    return np.array(P_hybrid_forecast_usd), np.array(P_gm_forecast_usd), P_actual_test, data_X0_prices_train

# =========================================================================
# === PHẦN 2: THỰC THI CHÍNH & HIỂN THỊ KẾT QUẢ ===
# =========================================================================

# Áp dụng các tinh chỉnh tối ưu mới
P_hybrid_forecast_usd, P_gm_forecast_usd, P_actual_test, P_train_usd = hybrid_gm_lstm_forecast_actual(
    data_X0_prices_full, 
    train_len=train_len,
    forecast_steps=forecast_steps, 
    lookback=50, # Lookback lớn hơn
    epochs=600  # Epochs cao hơn
)

# --- TÍNH TOÁN ĐỘ LỖI ---
mse_gm, mae_gm, mape_gm = calculate_metrics(P_actual_test, P_gm_forecast_usd)
mse_hybrid, mae_hybrid, mape_hybrid = calculate_metrics(P_actual_test, P_hybrid_forecast_usd)

error_results = pd.DataFrame({
    'Chỉ số': ['MSE', 'MAE', 'MAPE (%)'],
    'GM(1,1)': [f'{mse_gm:.4f}', f'{mae_gm:.4f}', f'{mape_gm:.4f}'],
    'GM + LSTM': [f'{mse_hybrid:.4f}', f'{mae_hybrid:.4f}', f'{mape_hybrid:.4f}']
})
print("\n## 📊 Bảng So sánh Chỉ số Độ lỗi (Sau khi chuyển sang Chuẩn hóa Giá)")
print(error_results.to_markdown(index=False))

# --- XỬ LÝ NGÀY THÁNG VÀ VẼ BIỂU ĐỒ ---
history_length_plot = train_len
P_past_plot = P_train_usd[-history_length_plot:] 

all_dates_indices = np.arange(train_len - history_length_plot, train_len + forecast_steps)
all_dates = dates_full[all_dates_indices]

# Xử lý trường hợp dates_full là mảng NumPy (Datetime)
try:
    all_dates_str = [pd.to_datetime(d).strftime('%m-%d') for d in all_dates]
except AttributeError:
    all_dates_str = [pd.to_datetime(d).strftime('%m-%d') for d in all_dates]

x_points = np.arange(history_length_plot + forecast_steps)
x_forecast = np.arange(history_length_plot - 1, history_length_plot + forecast_steps)

P_actual_test_plot = np.concatenate(([P_past_plot[-1]], P_actual_test))
P_gm_full = np.concatenate(([P_past_plot[-1]], P_gm_forecast_usd))
P_hybrid_full = np.concatenate(([P_past_plot[-1]], P_hybrid_forecast_usd))

plt.figure(figsize=(14, 6))

plt.plot(x_points[:history_length_plot], P_past_plot, marker='o', linestyle='-', color='blue', label='Giá Gốc (Train)', zorder=2)
plt.scatter(history_length_plot - 1, P_past_plot[-1], color='black', marker='o', s=100, zorder=5, label=f'P(n-5) - {all_dates_str[history_length_plot - 1]}')

plt.plot(x_forecast, P_actual_test_plot, marker='s', linestyle='-', color='purple', label='Giá Gốc (Test/Actual)', zorder=2)

plt.plot(x_forecast, P_gm_full, marker='^', linestyle=':', color='orange', label='Dự báo GM(1,1)', zorder=3)
plt.plot(x_forecast, P_hybrid_full, marker='x', linestyle='--', color='red', label='Dự báo Lai GM + LSTM', zorder=4)

plt.title('Biểu đồ So sánh Dự báo và Giá Gốc (Test Set) - Mô hình Giá Chuẩn hóa')
plt.xlabel('Ngày tháng')
plt.ylabel('Giá Đóng cửa (USD)')

plt.xticks(x_points, all_dates_str, rotation=45, ha='right')
plt.xlim(x_points.min(), x_points.max()) 

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n## 📋 Bảng So sánh Giá Gốc và Dự báo")
comparison_table = pd.DataFrame({
    'Ngày': all_dates_str[history_length_plot:],
    'Giá Gốc (Actual)': P_actual_test.round(4),
    'GM(1,1) Dự báo': P_gm_forecast_usd.round(4),
    'Lai GM+LSTM Dự báo': P_hybrid_forecast_usd.round(4)
})
print(comparison_table.to_markdown(index=False))