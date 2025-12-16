import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import time
import os
import glob 

# --- THAM SỐ CẤU HÌNH TỐI ƯU HÓA (DÀNH CHO DỮ LIỆU 5 NĂM) ---
LOOK_BACK = 60          
EPOCHS = 300           
BATCH_SIZE = 128              
TRAIN_RATIO = 0.7       
VAL_RATIO = 0.2       
LEARNING_RATE = 0.0001  
WEIGHT_DECAY = 1e-4     
PATIENCE = 30           
FORECAST_LENGTH = 90 
# === THAM SỐ LÀM MƯỢT DỮ LIỆU ===
EMA_WINDOW = 10          
# ------------------------------------

# SỬ DỤNG DICTIONARY CUNG CẤP
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

FEATURE_COLS = ['Close', 'High', 'Low', 'Open', 'Volume']
TARGET_COLUMN_INDEX = FEATURE_COLS.index('Close') 
INPUT_SIZE = len(FEATURE_COLS)

# THAM SỐ KIẾN TRÚC MÔ HÌNH
HIDDEN_SIZE = 256
NUM_LAYERS = 3 
OUTPUT_SIZE = 1

## 1. THIẾT LẬP CƠ SỞ VÀ MÔ HÌNH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Đang sử dụng thiết bị: {device}")

# Hàm tạo tập dữ liệu cửa sổ (Giữ nguyên)
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

# Định nghĩa Mô hình LSTM (Giữ nguyên)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :] 
        
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# --- VÒNG LẶP XỬ LÝ NHIỀU FILE ---
all_results = {}
all_close_data = {} 
total_start_time = time.time()

for ticker, file_path in COMPANY_FILES.items():
    print(f"\n=======================================================")
    print(f"🚀 BẮT ĐẦU XỬ LÝ MÃ: {ticker}")
    print(f"=======================================================")
    file_start_time = time.time()
    
    try:
        # TẢI DỮ LIỆU
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df.dropna(inplace=True)
        
        # Lưu dữ liệu Close gốc (CẦN LƯU TRƯỚC KHI LÀM MƯỢT)
        all_close_data[ticker] = df['Close'].copy()
        
        if len(df) < LOOK_BACK + 1:
            print(f"❌ Dữ liệu quá ngắn ({len(df)} dòng). Bỏ qua mã này.")
            continue
            
        # === BƯỚC MỚI: TÍNH VÀ ÁP DỤNG EMA ĐỂ LÀM MƯỢT ===
        print(f"Áp dụng EMA {EMA_WINDOW} ngày để làm mượt dữ liệu...")
        
        for col in ['Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                # Tính EMA và ghi đè lên cột gốc
                df[col] = df[col].ewm(span=EMA_WINDOW, adjust=False).mean()
        
        # Loại bỏ các hàng NaN được tạo ra do tính EMA (các hàng đầu tiên)
        df.dropna(inplace=True)
        # ========================================================
            
        data_to_use = df[FEATURE_COLS].to_numpy()
        
        # CHUẨN HÓA DỮ LIỆU
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_to_use) 
        
        # CHUẨN BỊ DATALOADER (Chia Train/Val/Test)
        X, Y = create_sequences(scaled_data, LOOK_BACK, TARGET_COLUMN_INDEX)
        
        # Cập nhật cách chia dữ liệu với VAL_RATIO mới
        train_size = int(len(X) * TRAIN_RATIO)
        val_size = int(len(X) * VAL_RATIO) 
        
        X_train = X[:train_size]
        Y_train = Y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        Y_val = Y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        Y_test = Y[train_size + val_size:]
        
        train_dataset = TimeSeriesDataset(X_train, Y_train)
        val_dataset = TimeSeriesDataset(X_val, Y_val)
        test_dataset = TimeSeriesDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"Kích thước tập: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH
        model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"Bắt đầu huấn luyện với {EPOCHS} epochs tối đa...")
        
        # Vòng lặp huấn luyện với Dừng sớm
        for epoch in range(EPOCHS):
            model.train()
            train_loss_sum = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # Có thể thêm Gradient Clipping ở đây nếu cần:
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer.step()
                train_loss_sum += loss.item()
            
            avg_train_loss = train_loss_sum / len(train_loader)

            model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss_sum += loss.item()
            
            avg_val_loss = val_loss_sum / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"Dừng sớm tại Epoch {epoch+1} do Validation Loss không cải thiện trong {PATIENCE} epoch.")
                break

        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # DỰ ĐOÁN VÀ ĐÁNH GIÁ TRÊN TẬP TEST CUỐI CÙNG
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
        
        # ĐẢO NGƯỢC CHUẨN HÓA (trên giá đã làm mượt)
        dummy_array_pred = np.zeros((len(predictions), INPUT_SIZE))
        dummy_array_pred[:, TARGET_COLUMN_INDEX] = predictions.flatten()
        predicted_prices_smoothed = scaler.inverse_transform(dummy_array_pred)[:, TARGET_COLUMN_INDEX]
        
        dummy_array_actual = np.zeros((len(actuals), INPUT_SIZE))
        dummy_array_actual[:, TARGET_COLUMN_INDEX] = actuals.flatten()
        actual_prices_smoothed = scaler.inverse_transform(dummy_array_actual)[:, TARGET_COLUMN_INDEX]
        
        # Tính RMSE và MAE trên giá đã làm mượt
        rmse = np.sqrt(np.mean(((predicted_prices_smoothed - actual_prices_smoothed) ** 2)))
        mae = np.mean(np.abs(predicted_prices_smoothed - actual_prices_smoothed)) 
        
        file_end_time = time.time()
        time_taken = file_end_time - file_start_time
        
        # LƯU KẾT QUẢ
        test_dates = df.index[len(df) - len(actual_prices_smoothed):]
        actual_prices_raw = all_close_data[ticker].loc[test_dates].values
        
        all_results[ticker] = {
            'RMSE': rmse, 
            'MAE': mae, 
            'Time (s)': time_taken,
            'Actuals_Smoothed': actual_prices_smoothed, 
            'Predictions_Smoothed': predicted_prices_smoothed,
            'Actuals_Raw': actual_prices_raw,
            'Dates': test_dates
        }
        print(f"✅ HOÀN TẤT MÃ: {ticker} | RMSE (Test-Smoothed): {rmse:.4f} | MAE (Test-Smoothed): {mae:.4f} | Thời gian: {time_taken:.2f} giây")
        
    except FileNotFoundError:
        print(f"❌ LỖI FILE: Không tìm thấy file tại đường dẫn: {file_path}")
    except KeyError as e:
        print(f"❌ LỖI DỮ LIỆU trong {ticker}: Không tìm thấy cột {e}. Vui lòng kiểm tra tiêu đề file.")
    except Exception as e:
        print(f"❌ LỖI BẤT THƯỜNG trong {ticker}: {e}")

# --- KẾT QUẢ TỔNG HỢP CUỐI CÙNG ---
print("\n\n#######################################################")
print("TỔNG HỢP KẾT QUẢ DỰ BÁO CÁC MÃ CỔ PHIẾU")
print("#######################################################")

final_results_df = pd.DataFrame({
    ticker: {'RMSE': res['RMSE'], 'MAE': res['MAE'], 'Time (s)': res['Time (s)']} 
    for ticker, res in all_results.items() if 'RMSE' in res
}).T
final_results_df.index.name = 'Ticker'
print(final_results_df.to_string(float_format='%.4f'))
total_end_time = time.time()
print(f"\n🔥 TỔNG THỜI GIAN XỬ LÝ BATCH: {total_end_time - total_start_time:.2f} giây")

# --- TRỰC QUAN HÓA TỔNG HỢP CÁC BIỂU ĐỒ CON (Di chuyển Legend ra ngoài) ---
if all_results:
    tickers = list(all_results.keys())
    num_tickers = len(tickers)
    
    rows = int(np.ceil(num_tickers / 3))
    cols = 3
    
    # 1. Gán plt.figure() cho biến fig và axes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Thiết lập tiêu đề chung
    fig.suptitle(f'Dự báo Giá Cổ phiếu LSTM - {FORECAST_LENGTH} Ngày Cuối (Dữ liệu Đã làm Mượt)', fontsize=18, fontweight='bold')

    lines_and_labels = [] 

    for i, ticker in enumerate(tickers):
        res = all_results[ticker]
        ax = axes[i] 
        
        # --- Chuẩn bị Dữ liệu ---
        N = FORECAST_LENGTH
        actual_prices_raw = res['Actuals_Raw'][-N:] 
        predicted_prices_smoothed = res['Predictions_Smoothed'][-N:]
        plot_dates = res['Dates'][-N:]
        history_look_back = LOOK_BACK * 3
        
        if ticker in all_close_data and not plot_dates.empty:
            start_date_forecast = plot_dates[0]
            history_data = all_close_data[ticker][:start_date_forecast].iloc[-history_look_back:]
            history_dates = history_data.index
        else:
            history_data = pd.Series([]) 
            history_dates = pd.Index([])

        # 1. Vẽ lịch sử GỐC
        if not history_data.empty:
            l1, = ax.plot(history_dates, np.array(history_data.values), label='Lịch sử (Train/Val)', color='grey', linewidth=1.0, alpha=0.7)
            
        # 2. Vẽ giá trị THỰC TẾ GỐC (Màu xanh)
        l2, = ax.plot(plot_dates, np.array(actual_prices_raw), label='Thực tế (Test)', color='#1f77b4', linewidth=1.5)

        # 3. Vẽ giá trị Dự đoán ĐÃ LÀM MƯỢT (Màu đỏ)
        l3, = ax.plot(plot_dates, np.array(predicted_prices_smoothed), label='Dự đoán (LSTM)', color='#d62728', linestyle='--', alpha=0.8, linewidth=1.8)
        
        # 4. Điểm bắt đầu dự đoán (đánh dấu)
        if not plot_dates.empty:
            ax.scatter(plot_dates[0], actual_prices_raw[0], color='#d62728', marker='o', s=40, zorder=5) 

        # 5. Thu thập đối tượng line và label (chỉ cần lấy 1 lần)
        if i == 0:
            lines_and_labels = [(l1, 'Lịch sử (Train/Val)'), (l2, 'Thực tế (Test)'), (l3, 'Dự đoán (LSTM)')]

        # Định dạng biểu đồ con
        ax.set_title(f'{ticker} (RMSE: {res["RMSE"]:.2f}, MAE: {res["MAE"]:.2f})', fontsize=12)
        ax.set_ylabel('Giá Đóng Cửa', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Tối ưu hóa Trục X
        date_format = mdates.DateFormatter('%m/%d/%y')
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        
        # Ẩn các trục con không sử dụng (nếu num_tickers không chia hết cho cols)
        if i >= num_tickers:
            fig.delaxes(ax) 

    # 6. TẠO CHÚ THÍCH CHUNG (Sử dụng fig.legend)
    if lines_and_labels:
        lines, labels = zip(*lines_and_labels)
        fig.legend(lines, labels, loc='lower center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.01), frameon=True)


    # 7. Áp dụng các điều chỉnh bố cục (để Legend và suptitle không chồng chéo)
    
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    plt.subplots_adjust(
        wspace=0.35, 
        hspace=0.55,  
        bottom=0.10   
    )
    plt.show()
