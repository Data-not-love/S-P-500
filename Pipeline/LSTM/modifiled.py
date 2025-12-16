import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import os
import glob 

# --- THAM SỐ CẤU HÌNH ĐÃ TỐI ƯU HÓA ---
LOOK_BACK = 60         
EPOCHS = 100
BATCH_SIZE = 128       
TRAIN_RATIO = 0.8
LEARNING_RATE = 0.0005
MODEL_TYPE = 'LSTM'    
NUM_WORKERS = 2        
# ----------------------------------------

# DICTIONARY CUNG CẤP ĐƯỜNG DẪN FILE DỮ LIỆU
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
FEATURE_COLS = ['Close', 'High', 'Low', 'Open', 'Volume']
TARGET_COLUMN_INDEX = FEATURE_COLS.index('Close') 
INPUT_SIZE = len(FEATURE_COLS)
HIDDEN_SIZE = 256
NUM_LAYERS = 3 
OUTPUT_SIZE = 1


## ĐỊNH NGHĨA HÀM VÀ CLASS

# Hàm tạo tập dữ liệu cửa sổ
def create_sequences(data, look_back, target_index):
    sequences, targets = [], []
    for i in range(len(data) - look_back):
        sequences.append(data[i:i + look_back, :]) 
        targets.append(data[i + look_back, target_index]) 
    return np.array(sequences), np.array(targets).reshape(-1, 1) 
    
class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Định nghĩa Mô hình LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# =======================================================
# KHỐI THỰC THI CHÍNH 
# =======================================================
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Đang sử dụng thiết bị: {device}")
    
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler() 
        print("⚡ Đã khắc phục lỗi cú pháp PyTorch. Đang sử dụng Automatic Mixed Precision (AMP).")
    else:
        scaler = None
        print("❌ AMP chỉ khả dụng với CUDA. Huấn luyện sẽ chạy trên CPU.")


    all_results = {}
    total_start_time = time.time()

    for ticker, file_path in COMPANY_FILES.items():
        print(f"\n=======================================================")
        print(f"🚀 BẮT ĐẦU XỬ LÝ MÃ: {ticker} (Model: {MODEL_TYPE})")
        print(f"=======================================================")
        file_start_time = time.time()
        
        try:
            # TẢI DỮ LIỆU
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            df.dropna(inplace=True)
            if len(df) < LOOK_BACK + 1:
                print(f"❌ Dữ liệu quá ngắn ({len(df)} dòng). Bỏ qua mã này.")
                continue
                
            data_to_use = df[FEATURE_COLS].to_numpy()
            
            # CHUẨN HÓA DỮ LIỆU
            min_max_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = min_max_scaler.fit_transform(data_to_use) 
            
            # CHUẨN BỊ DATALOADER
            X, Y = create_sequences(scaled_data, LOOK_BACK, TARGET_COLUMN_INDEX)
            train_size = int(len(X) * TRAIN_RATIO)
            X_train, X_test = X[:train_size], X[train_size:]
            Y_train, Y_test = Y[:train_size], Y[train_size:]
            
            train_dataset = TimeSeriesDataset(X_train, Y_train)
            test_dataset = TimeSeriesDataset(X_test, Y_test)
            
            # TỐI ƯU I/O
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                      num_workers=NUM_WORKERS, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=True)
            
            # KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH
            model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            print(f"Bắt đầu huấn luyện {EPOCHS} epochs...")
            model.train()
            
            # VÒNG LẶP HUẤN LUYỆN VỚI AMP (NẾU CÓ CUDA)
            for epoch in range(EPOCHS):
                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    if scaler:
                        # Sử dụng torch.autocast để tránh lỗi "not exported"
                        with torch.autocast(device_type='cuda', dtype=torch.float16): 
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Chạy bình thường trên CPU
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
            
            # DỰ ĐOÁN VÀ ĐÁNH GIÁ
            model.eval() 
            predictions, actuals = [], []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    predictions.append(outputs.cpu().numpy())
                    actuals.append(targets.cpu().numpy())
            
            predictions = np.concatenate(predictions)
            actuals = np.concatenate(actuals)
            
            # ĐẢO NGƯỢC CHUẨN HÓA (Dùng min_max_scaler gốc)
            dummy_array = np.zeros((len(predictions), INPUT_SIZE))
            dummy_array[:, TARGET_COLUMN_INDEX] = predictions.flatten()
            predicted_prices = min_max_scaler.inverse_transform(dummy_array)[:, TARGET_COLUMN_INDEX]
            
            dummy_array = np.zeros((len(actuals), INPUT_SIZE))
            dummy_array[:, TARGET_COLUMN_INDEX] = actuals.flatten()
            actual_prices = min_max_scaler.inverse_transform(dummy_array)[:, TARGET_COLUMN_INDEX]
            
            rmse = np.sqrt(np.mean(((predicted_prices - actual_prices) ** 2)))
            
            file_end_time = time.time()
            time_taken = file_end_time - file_start_time
            
            # LƯU KẾT QUẢ
            all_results[ticker] = {
                'RMSE': rmse, 
                'Time (s)': time_taken,
                'Actuals': actual_prices,
                'Predictions': predicted_prices,
                'Dates': df.index[len(df) - len(actual_prices):]
            }
            print(f"✅ HOÀN TẤT MÃ: {ticker} | RMSE: {rmse:.4f} | Thời gian: {time_taken:.2f} giây")
            
            
            ## 6. TRỰC QUAN HÓA KẾT QUẢ RIÊNG LẺ 📊
            plt.figure(figsize=(14, 7))
            plt.plot(all_results[ticker]['Dates'], actual_prices, label='Giá trị Thực tế', color='blue')
            plt.plot(all_results[ticker]['Dates'], predicted_prices, label='Giá trị Dự đoán (LSTM)', color='red', alpha=0.7)
            plt.title(f'Dự báo Giá Cổ phiếu {ticker} - RMSE: {rmse:.2f}')
            plt.xlabel('Ngày')
            plt.ylabel('Giá Đóng cửa (USD)')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except FileNotFoundError:
            print(f"❌ LỖI FILE: Không tìm thấy file tại đường dẫn: {file_path}")
        except KeyError as e:
            print(f"❌ LỖI DỮ LIỆU trong {ticker}: Không tìm thấy cột {e}. Vui lòng kiểm tra tiêu đề file.")
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"❌ LỖI BỘ NHỚ GPU: Lỗi CUDA Out of Memory trong {ticker}. Vui lòng GIẢM BATCH_SIZE (ví dụ: về 64) HOẶC GIẢM HIDDEN_SIZE.")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                print(f"❌ LỖI BẤT THƯỜNG trong {ticker}: {e}")
        except Exception as e:
            print(f"❌ LỖI BẤT THƯỜNG trong {ticker}: {e}")

    # --- KẾT QUẢ TỔNG HỢP CUỐI CÙNG ---
    print("\n\n#######################################################")
    print("TỔNG HỢP KẾT QUẢ DỰ BÁO CÁC MÃ CỔ PHIẾU")
    print("#######################################################")
    
    final_results_df = pd.DataFrame({
        ticker: {'RMSE': res['RMSE'], 'Time (s)': res['Time (s)']}
        for ticker, res in all_results.items() if 'RMSE' in res
    }).T
    final_results_df.index.name = 'Ticker'
    print(final_results_df.to_string(float_format='%.4f'))

    total_end_time = time.time()
    print(f"\n🔥 TỔNG THỜI GIAN XỬ LÝ BATCH: {total_end_time - total_start_time:.2f} giây")

    # --- TRỰC QUAN HÓA TỔNG HỢP (So sánh RMSE) ---
    if len(final_results_df) > 0:
        plt.figure(figsize=(10, 6))
        final_results_df['RMSE'].sort_values().plot(kind='bar', color='skyblue')
        plt.title('So sánh RMSE dự báo LSTM giữa các Mã Cổ phiếu')
        plt.ylabel('RMSE (USD)')
        plt.xlabel('Mã Cổ phiếu (Ticker)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()