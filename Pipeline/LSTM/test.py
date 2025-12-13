import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

# --- THAM SỐ CẤU HÌNH ĐÃ CẬP NHẬT ---
LOOK_BACK = 120
EPOCHS = 300
BATCH_SIZE = 64 
TRAIN_RATIO = 0.8
FILE_NAME = 'D:\\S-P-500-development\\raw data\\3M\\MMM 5y.csv'
# Dựa trên file của bạn: CHỌN CÁC CỘT TIÊU CHUẨN ĐỂ HUẤN LUYỆN
FEATURE_COLS = ['Close', 'High', 'Low', 'Open', 'Volume']
# Mục tiêu dự đoán là 'Close', là index 0 trong list FEATURE_COLS
TARGET_COLUMN_INDEX = FEATURE_COLS.index('Close') 
INPUT_SIZE = len(FEATURE_COLS) # INPUT_SIZE = 5
HIDDEN_SIZE = 256
NUM_LAYERS = 3 
OUTPUT_SIZE = 1
LEARNING_RATE = 0.0005
# ----------------------------------------

## 1. TẢI DỮ LIỆU VÀ THIẾT LẬP DEVICE 🚀

# Thiết lập thiết bị (GPU nếu có CUDA, ngược lại là CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Đang sử dụng thiết bị: {device}")
if device.type == 'cuda':
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")

try:
    # Lấy cột 'Date' làm Index
    df = pd.read_csv(FILE_NAME, index_col='Date', parse_dates=True)
    
    # CHỌN CÁC CỘT ĐÃ ĐỊNH NGHĨA
    data_to_use = df[FEATURE_COLS].to_numpy() 

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{FILE_NAME}'. Vui lòng kiểm tra lại tên file.")
    exit()
except KeyError as e:
    # Báo lỗi chính xác tên cột bị thiếu nếu có
    print(f"LỖI: Không tìm thấy cột {e} trong file. Vui lòng kiểm tra FEATURE_COLS có khớp với header không.")
    exit()

# Chuẩn hóa dữ liệu (Áp dụng cho tất cả các cột)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_to_use) 

## 2. CHUẨN BỊ DATALOADER 🖼️

# Hàm tạo tập dữ liệu cửa sổ cho Đa biến
def create_sequences(data, look_back, target_index):
    sequences, targets = [], []
    for i in range(len(data) - look_back):
        sequences.append(data[i:i + look_back, :]) 
        targets.append(data[i + look_back, target_index]) 
    return np.array(sequences), np.array(targets).reshape(-1, 1) 
    
X, Y = create_sequences(scaled_data, LOOK_BACK, TARGET_COLUMN_INDEX)

# Chia tập huấn luyện/kiểm tra và tạo DataLoader
train_size = int(len(X) * TRAIN_RATIO)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_dataset = TimeSeriesDataset(X_train, Y_train)
test_dataset = TimeSeriesDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

## 3. XÂY DỰNG MÔ HÌNH LSTM BẰNG PYTORCH 🧠

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # input_size = 5 (các features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

## 4. HUẤN LUYỆN MÔ HÌNH TRÊN CUDA/GPU 🚀

start_time = time.time()
model.train() 

for epoch in range(EPOCHS):
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}')

end_time = time.time()
print(f"\n✅ Hoàn tất huấn luyện. Tổng thời gian: {end_time - start_time:.2f} giây")

## 5. DỰ ĐOÁN VÀ ĐÁNH GIÁ 🔮

model.eval() 

predictions = []
actuals = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        predictions.append(outputs.cpu().numpy())
        actuals.append(targets.cpu().numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# Đảo ngược chuẩn hóa (chỉ cột Close)
# Cần tạo mảng giả với kích thước 5 cột để MinMaxScaler có thể đảo ngược
dummy_array = np.zeros((len(predictions), INPUT_SIZE))
dummy_array[:, TARGET_COLUMN_INDEX] = predictions.flatten()
predicted_prices = scaler.inverse_transform(dummy_array)[:, TARGET_COLUMN_INDEX]

dummy_array = np.zeros((len(actuals), INPUT_SIZE))
dummy_array[:, TARGET_COLUMN_INDEX] = actuals.flatten()
actual_prices = scaler.inverse_transform(dummy_array)[:, TARGET_COLUMN_INDEX]


# Tính toán RMSE
rmse = np.sqrt(np.mean(((predicted_prices - actual_prices) ** 2)))
print(f"\nRMSE trên tập kiểm tra: {rmse:.2f} (Độ lệch trung bình so với thực tế)")

# IN RA GIÁ TRỊ
test_index = df.index[len(df) - len(actual_prices):]
results_df = pd.DataFrame({
    'Date': test_index,
    'Actual_Price': actual_prices,
    'Predicted_Price': predicted_prices
})

print("\n📈 20 Giá trị Dự đoán và Thực tế đầu tiên trong Tập Kiểm tra:")
print(results_df.head(20).to_string(index=False, float_format='%.2f'))

## 6. TRỰC QUAN HÓA KẾT QUẢ 📊

plt.figure(figsize=(14, 7))
plt.plot(test_index, actual_prices, label='Giá trị Thực tế', color='blue')
plt.plot(test_index, predicted_prices, label='Giá trị Dự đoán (Multi-variate LSTM)', color='red', alpha=0.7)
plt.title(f'Dự báo Giá Cổ phiếu MMM bằng PyTorch LSTM (Multi-variate)')
plt.xlabel('Ngày')
plt.ylabel('Giá Đóng cửa (USD)')
plt.legend()
plt.grid(True)
plt.show()