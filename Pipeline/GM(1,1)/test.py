import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import io
import matplotlib.pyplot as plt

# --- 1. Định nghĩa Lớp GM(1,1) (Dùng NumPy - Ổn định) ---
class GreyModelGM11_NumPy:
    def __init__(self):
        self.a = None
        self.b = None
        self.X0 = None 
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
        
        X0_fitted = np.insert(X0_pred_all[:n-1], 0, self.X0_start)
        X0_forecast = X0_pred_all[n-1:n-1+steps]
        X0_all_fitted = X0_pred_all[:n] # Tất cả các điểm fitted (bao gồm X(0)(1))
        
        return X0_fitted, X0_forecast, X0_all_fitted 

# --- 2. Định nghĩa Mô hình LSTM (PyTorch - Đã sửa lỗi) ---
class LSTMResidualModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMResidualModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x) 
        out = self.fc(out[:, -1, :])
        return out

# --- 3. Hàm tạo Dữ liệu chuỗi thời gian cho LSTM ---
def create_dataset(data, lookback=1):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), 0])
        Y.append(data[i + lookback, 0])
    return np.array(X), np.array(Y)

# --- 4. Quá trình Lai (Hybrid GM-LSTM) ---
def hybrid_gm_lstm_forecast(data_X0_prices, lookback=5, forecast_steps=5, epochs=100):
    
    # 4.1. Biến đổi dữ liệu gốc
    log_returns = np.log(data_X0_prices[1:] / data_X0_prices[:-1])
    offset = np.abs(np.min(log_returns)) + 0.001
    X0_transformed = log_returns + offset
    
    # 4.2. Huấn luyện GM(1,1) và tính toán Residuals
    gm11_model = GreyModelGM11_NumPy()
    gm11_model.fit(X0_transformed)
    
    X0_fitted_gm_plot, X0_forecast_gm, X0_all_fitted = gm11_model.get_fitted_and_forecast(forecast_steps)
    
    # Residuals (Phần dư) = Thực tế - Fitted của GM
    residuals_actual = X0_transformed - X0_fitted_gm_plot
    
    # 4.3. Tiền xử lý Residuals cho LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    residuals_scaled = scaler.fit_transform(residuals_actual.reshape(-1, 1))
    X_res, Y_res = create_dataset(residuals_scaled, lookback)
    
    X_res_torch = torch.from_numpy(X_res).float().unsqueeze(-1)
    Y_res_torch = torch.from_numpy(Y_res).float().unsqueeze(-1)
    
    # 4.4. Huấn luyện LSTM trên Residuals
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 1
    hidden_size = 32
    output_size = 1
    num_layers = 2
    
    lstm_model = LSTMResidualModel(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    
    X_res_torch = X_res_torch.to(device)
    Y_res_torch = Y_res_torch.to(device)
    
    for epoch in range(epochs):
        lstm_model.train()
        outputs = lstm_model(X_res_torch)
        loss = criterion(outputs, Y_res_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 4.5. Dự báo Lai (Hybrid Forecast)
    lstm_model.eval()
    current_input = residuals_scaled[-lookback:].copy() # Dùng bản sao
    lstm_residual_forecast = []
    
    for _ in range(forecast_steps):
        input_tensor = torch.from_numpy(current_input.reshape(1, lookback, 1)).float().to(device)
        with torch.no_grad():
            predicted_scaled = lstm_model(input_tensor).detach().cpu().numpy().flatten()[0]
        
        lstm_residual_forecast.append(predicted_scaled)
        current_input = np.append(current_input[1:], predicted_scaled)[-lookback:]
    
    # Đảo ngược Scaling của Residuals dự báo
    residual_forecast_orig = scaler.inverse_transform(np.array(lstm_residual_forecast).reshape(-1, 1)).flatten()
    
    # Residuals fitted (Để vẽ biểu đồ)
    X_res_fitted = lstm_model(X_res_torch).detach().cpu().numpy().flatten()
    residuals_fitted_orig = scaler.inverse_transform(X_res_fitted.reshape(-1, 1)).flatten()
    
    # Dự báo Lai = GM Forecast + LSTM Residual Forecast
    hybrid_forecast_transformed = X0_forecast_gm + residual_forecast_orig
    
    # Đảo ngược Biến đổi về Giá cổ phiếu USD
    P_current = data_X0_prices[0] # Giá mới nhất
    P_hybrid_forecast_usd = []
    
    for x in hybrid_forecast_transformed:
        log_return_pred = x - offset
        P_next = P_current * np.exp(log_return_pred)
        P_hybrid_forecast_usd.append(P_next)
        P_current = P_next
        
    # Giá GM(1,1) đơn thuần dự báo
    P_current_gm = data_X0_prices[0]
    P_gm_forecast_usd = []
    for x in X0_forecast_gm:
        log_return_pred = x - offset
        P_next = P_current_gm * np.exp(log_return_pred)
        P_gm_forecast_usd.append(P_next)
        P_current_gm = P_next
        
    return P_hybrid_forecast_usd, P_gm_forecast_usd, residuals_actual, residuals_fitted_orig, data_X0_prices, gm11_model.a, gm11_model.b

# --- 5. Thực thi và Vẽ Biểu đồ ---

# Tải dữ liệu (Sử dụng 20 điểm cuối)
csv_content = """Price,Close,High,Low,Open,Volume,Date
117.07459259033203,117.07459259033203,117.49005054216046,114.80346120514672,114.9765665790167,1959168,2020-10-15
118.38330078125,118.38330078125,119.14495700201921,117.17156291350676,117.67010497701581,2964406,2020-10-16
117.40003967285156,117.40003967285156,119.33880971576845,116.92918396633665,118.4178977362882,2284240,2020-10-19
120.0,120.0,121.0,119.0,120.0,1500000,2020-10-20
121.5,121.5,122.0,120.5,121.0,1600000,2020-10-21
122.1,122.1,123.0,121.0,121.5,1700000,2020-10-22
123.5,123.5,124.0,122.5,123.0,1800000,2020-10-23
125.0,125.0,125.5,124.0,124.5,1900000,2020-10-26
126.5,126.5,127.0,125.5,126.0,2000000,2020-10-27
127.8,127.8,128.0,127.0,127.0,2100000,2020-10-28
129.0,129.0,129.5,128.0,128.5,2200000,2020-10-29
130.5,130.5,131.0,129.5,130.0,2300000,2020-10-30
131.9,131.9,132.5,131.0,131.5,2400000,2020-11-02
133.0,133.0,133.5,132.0,132.5,2500000,2020-11-03
134.5,134.5,135.0,133.5,134.0,2600000,2020-11-04
136.0,136.0,136.5,135.0,135.5,2700000,2020-11-05
137.5,137.5,138.0,136.5,137.0,2800000,2020-11-06
139.0,139.0,139.5,138.0,138.5,2900000,2020-11-09
140.5,140.5,141.0,139.5,140.0,3000000,2020-11-10
155.17999267578125,155.17999267578125,155.8800048828125,154.67999267578125,155.0,2270900,2025-09-30
"""
df = pd.read_csv(io.StringIO(csv_content), index_col=False, on_bad_lines='skip')
df = df[['Close']]

close_prices = df['Close'].values[::-1] # Đảo ngược: Mới nhất -> Cũ nhất
data_X0_prices = close_prices[:20] 

if len(data_X0_prices) < 20:
    data_X0_prices = close_prices # Sử dụng tất cả nếu ít hơn 20


P_hybrid_forecast_usd, P_gm_forecast_usd, residuals_actual, residuals_fitted_orig, data_X0_prices, a_gm, b_gm = hybrid_gm_lstm_forecast(
    data_X0_prices, 
    lookback=5, 
    forecast_steps=5, 
    epochs=100
)

# --- VẼ BIỂU ĐỒ 1: RESIDUALS (PHẦN DƯ) ---
lookback = 5
X_axis_actual = np.arange(1, len(residuals_actual) + 1)
X_axis_fitted = np.arange(lookback + 1, len(residuals_actual) + 1)

plt.figure(figsize=(12, 5))
plt.plot(X_axis_actual, residuals_actual, marker='o', linestyle='-', color='blue', label='Residuals Thực tế (GM(1,1))')
plt.plot(X_axis_fitted, residuals_fitted_orig, marker='x', linestyle='--', color='red', label='Residuals Fitted (LSTM)')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Biểu đồ 1: Residuals của GM(1,1) và Fitted bởi LSTM')
plt.xlabel('Bước Thời gian (k)')
plt.ylabel('Giá trị Phần dư')
plt.legend()
plt.grid(True)
plt.show() # Hiển thị biểu đồ 1

# --- VẼ BIỂU ĐỒ 2: SO SÁNH DỰ BÁO GIÁ ---
# Dữ liệu thực tế: 10 điểm cuối cùng
P_past_usd = data_X0_prices[:10][::-1] 
n_past = len(P_past_usd)

# Trục X
x_past = np.arange(n_past)
x_forecast = np.arange(n_past, n_past + 5)

plt.figure(figsize=(12, 6))
# 1. Giá Gốc
plt.plot(x_past, P_past_usd, marker='o', linestyle='-', color='blue', label='Giá Đóng cửa Gốc')

# 2. Dự báo GM(1,1) đơn thuần
plt.plot(x_forecast, P_gm_forecast_usd, marker='^', linestyle=':', color='orange', label='Dự báo GM(1,1)')

# 3. Dự báo Lai GM + LSTM
plt.plot(x_forecast, P_hybrid_forecast_usd, marker='x', linestyle='--', color='red', label='Dự báo Lai GM + LSTM')

# Đánh dấu điểm P(n)
plt.scatter(n_past - 1, P_past_usd[-1], color='black', marker='o', s=100, zorder=5, label='P(n) - Giá cuối')

plt.title('Biểu đồ 2: So sánh Dự báo Giá Đóng cửa (GM(1,1) vs GM + LSTM)')
plt.xlabel('Bước Thời gian')
plt.ylabel('Giá Đóng cửa (USD)')

# Nhãn trục X tùy chỉnh
all_labels = [f'P(n-{n_past-1-i})' for i in range(n_past)] + [f'P(n+{i+1})' for i in range(5)]
plt.xticks(np.arange(n_past + 5), all_labels, rotation=45, ha='right')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() # Hiển thị biểu đồ 2

print("\n--- KẾT QUẢ DỰ BÁO MÔ HÌNH LAI GM(1,1) + LSTM ---")
print(f"Giá đóng cửa cuối cùng (P(n)): {data_X0_prices[0]:.4f}")
print(f"Tham số GM(1,1) [a, b]: [{a_gm:.6f}, {b_gm:.6f}]")
print("-------------------------------------------------------")

for i, price in enumerate(P_hybrid_forecast_usd):
    print(f"Dự báo Giá Lai Ngày {i+1} (P(n+{i+1})): {price:.4f}")