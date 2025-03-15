import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import logging
from sklearn.preprocessing import StandardScaler

# ============================================================
# PART 1: Data Loading & Preprocessing (FFN 방식)
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)  # 예: 'data/MBD_VT_DATA'

    def load_data(self, noise_level: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        데이터 로드 및 전처리
        Args:
            noise_level: 예를 들어 'clean', 'noise_40%' 등
        Returns:
            time_train, time_test, time_valid, data_train, data_test, data_valid
            - time_*: (n, 1) 텐서 (시간 데이터)
            - data_*: (n, 3) 텐서 (가속도 데이터: Acc_TX@StRq3, Acc_TY@StRq3, Acc_TZ@StRq3)
        """
        try:
            file_path = self.base_path / noise_level
            datasets = {}
            for split in ['train', 'test', 'valid']:
                input_data = np.loadtxt(file_path / f"{noise_level}_input_{split}.txt", delimiter='\t', skiprows=1)
                output_data = np.loadtxt(file_path / f"{noise_level}_output_{split}.txt", delimiter='\t', skiprows=1)
                datasets[f"time_{split}"] = input_data.reshape(-1, 1)
                datasets[f"data_{split}"] = output_data
            return self._preprocess_data(datasets)
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise

    def _preprocess_data(self, datasets: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        데이터 전처리 및 텐서 변환  
        MBD 데이터는 3열(Acc_TX@StRq3, Acc_TY@StRq3, Acc_TZ@StRq3)로 구성되어 있습니다.
        여기서는 시간 데이터와 결합한 후, 시간 기준으로 정렬하고,  
        학습 데이터의 가속도에 대해 StandardScaler로 정규화(fit_transform)한 후,
        검증 및 테스트 데이터에는 동일한 scaler의 transform을 적용합니다.
        """
        processed_data = {}
        scaler = StandardScaler()
        # 학습 데이터 처리
        combined_train = np.concatenate((datasets["time_train"], datasets["data_train"]), axis=1)
        sorted_train = combined_train[np.argsort(combined_train[:, 0])]
        time_train = sorted_train[:, 0].reshape(-1, 1)
        train_acc = sorted_train[:, 1:]
        scaled_train_acc = scaler.fit_transform(train_acc)
        processed_data["time_train"] = torch.FloatTensor(time_train)
        processed_data["data_train"] = torch.FloatTensor(scaled_train_acc)
        
        # 검증 및 테스트 데이터 처리 (학습 데이터 scaler 사용)
        for split in ['test', 'valid']:
            combined = np.concatenate((datasets[f"time_{split}"], datasets[f"data_{split}"]), axis=1)
            sorted_data = combined[np.argsort(combined[:, 0])]
            time_split = sorted_data[:, 0].reshape(-1, 1)
            acc_split = sorted_data[:, 1:]
            scaled_acc = scaler.transform(acc_split)
            processed_data[f"time_{split}"] = torch.FloatTensor(time_split)
            processed_data[f"data_{split}"] = torch.FloatTensor(scaled_acc)
        
        return (processed_data["time_train"], processed_data["time_test"], 
                processed_data["time_valid"], processed_data["data_train"], 
                processed_data["data_test"], processed_data["data_valid"])

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    데이터를 시퀀스로 변환합니다.
    Args:
        data: (n, feature) 배열
        seq_length: 시퀀스 길이
    Returns:
        X: (n - seq_length, seq_length, feature)
        y: (n - seq_length, feature) → 다음 시점 값
    """
    sequences = [data[i:i+seq_length] for i in range(len(data) - seq_length)]
    targets = [data[i+seq_length] for i in range(len(data) - seq_length)]
    return np.array(sequences), np.array(targets)

def convert_to_tensor(X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    numpy 배열을 torch Tensor로 변환합니다.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

# ============================================================
# PART 2: LSTM Model Definition
# ============================================================
class LSTMPendulum(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        """
        Args:
            input_size: 입력 차원 (여기서는 3: 가속도 3채널)
            hidden_size: LSTM 은닉층 크기
            output_size: 출력 차원 (3: 다음 시점의 3채널 가속도)
            num_layers: LSTM 레이어 수
        """
        super(LSTMPendulum, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # 마지막 시퀀스 단계의 출력을 사용하여 fc layer 통과
        out = self.fc(lstm_out[:, -1, :])
        return out

# ============================================================
# PART 3: Training & Validation Loop for LSTM
# ============================================================
def train_model(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, scheduler,
                X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, 
                X_valid_tensor: torch.Tensor, y_valid_tensor: torch.Tensor,
                num_epochs: int, batch_size: int, noise_level: str) -> Tuple[list, list]:
    train_losses, valid_losses = [], []
    num_batches = int(np.ceil(len(X_train_tensor) / batch_size))
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        model.eval()
        with torch.no_grad():
            y_valid_pred = model(X_valid_tensor)
            valid_loss = criterion(y_valid_pred, y_valid_tensor)
        valid_losses.append(valid_loss.item())
        
        scheduler.step(valid_loss)
        
        # 최적 모델 저장
        if valid_loss.item() < best_valid_loss:
            best_valid_loss = valid_loss.item()
            best_model_path = f'example/trained_models/MBD_VT_LSTM_model_{noise_level}.pth'
            os.makedirs(Path(best_model_path).parent, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {valid_loss.item():.4f}')
    
    return train_losses, valid_losses

def plot_results(time_test: np.ndarray, test_data: np.ndarray, predictions: np.ndarray, 
                 seq_length: int, save_path: str):
    """
    결과 시각화 및 저장.
    test_data와 predictions는 (n,3) 배열 (각 채널: Acc_TX, Acc_TY, Acc_TZ)
    시퀀스 생성으로 인해 처음 seq_length만큼은 타깃이 없으므로 제외.
    """
    plt.figure(figsize=(15, 8))
    channel_labels = ['Acc_TX@StRq3', 'Acc_TY@StRq3', 'Acc_TZ@StRq3']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time_test[seq_length:], test_data[seq_length:, i], label=f'True {channel_labels[i]}', color='black')
        plt.plot(time_test[:], predictions[:, i], label=f'Predicted {channel_labels[i]}', linestyle='--',
                 color='red' if i==0 else ('blue' if i==1 else 'green'))
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Result plot saved to: {save_path}")

# ============================================================
# PART 4: Main Function for LSTM Training on MBD_VT_DATA (FFN 방식 데이터)
# ============================================================
def main():
    # 하이퍼파라미터 설정
    noise_level = 'clean'   # 예: 'clean', 'noise_40%', 등
    seq_length = 20         # 시퀀스 길이
    hidden_size = 128       # 성능 개선을 위해 hidden_size 증가
    num_layers = 3          # LSTM 레이어 수 증가
    num_epochs = 200
    learning_rate = 0.001
    batch_size = 32

    # FFN 방식으로 저장된 데이터 로드 (base path: data/MBD_VT_DATA)
    data_loader = DataLoader('data/MBD_VT_DATA')
    # 데이터 파일은 폴더 구조: data/MBD_VT_DATA/{noise_level}/{noise_level}_input_train.txt 등
    time_train, time_test, time_valid, data_train, data_test, data_valid = data_loader.load_data(noise_level)
    
    # 데이터는 이미 전처리되어 시간 (n,1)와 가속도 (n,3) 텐서로 리턴됩니다.
    # 여기서 LSTM의 입력은 시퀀스로 생성할 가속도 데이터만 사용합니다.
    # 따라서, 데이터를 numpy 배열로 변환한 후 시퀀스 생성
    data_train_np = data_train.numpy()  # (n_train, 3)
    data_valid_np = data_valid.numpy()  # (n_valid, 3)
    data_test_np  = data_test.numpy()   # (n_test, 3)
    
    # 시퀀스 생성: 입력은 seq_length 길이의 연속된 가속도, 타깃은 바로 다음 시점의 가속도
    X_train, y_train = create_sequences(data_train_np, seq_length)
    X_valid, y_valid = create_sequences(data_valid_np, seq_length)
    X_test, y_test   = create_sequences(data_test_np, seq_length)
    
    # 텐서 변환
    X_train_tensor, y_train_tensor = convert_to_tensor(X_train, y_train)
    X_valid_tensor, y_valid_tensor = convert_to_tensor(X_valid, y_valid)
    X_test_tensor, y_test_tensor   = convert_to_tensor(X_test, y_test)
    
    # LSTM 모델 초기화: 입력 크기는 3 (가속도 3채널), 출력 크기도 3
    input_size = 3
    output_size = 3
    model = LSTMPendulum(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    # 모델 학습
    train_losses, valid_losses = train_model(model, criterion, optimizer, scheduler,
                                              X_train_tensor, y_train_tensor,
                                              X_valid_tensor, y_valid_tensor,
                                              num_epochs, batch_size, noise_level)
    
    # 테스트 데이터 예측
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
    
    # 역정규화: StandardScaler로 정규화한 경우, 원래 스케일로 복원하려면 scaler.inverse_transform 필요
    # (여기서는 DataLoader 내부에서 정규화를 수행했으므로, 역정규화를 위해서는 동일한 scaler가 필요합니다.)
    # 예시에서는 역정규화 과정을 생략하고, 정규화된 값으로 플롯합니다.
    
    # 시퀀스 생성으로 인해 y_test의 길이는 원래 time_test보다 seq_length만큼 짧으므로,
    # 플롯 시 시간 데이터도 슬라이스해서 사용합니다.
    time_test_np = time_test.numpy().flatten()
    time_test_seq = time_test_np[seq_length:]
    
    # 결과 플롯 저장: 예시 경로 "example/figs/MBD_VT_LSTM_fig_clean.png"
    results_fig_path = f'example/figs/MBD_VT_LSTM_fig_{noise_level}.png'
    os.makedirs(Path(results_fig_path).parent, exist_ok=True)
    plot_results(time_test_seq, y_test, predictions, seq_length, results_fig_path)
    print(f"Result figure saved to {results_fig_path}")
    
    mse = np.mean((y_test - predictions) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")

if __name__ == '__main__':
    main()
