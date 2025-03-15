import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import logging

# ============================================================
# PART 1: Data Loading & Preprocessing Functions
# ============================================================
def data_loading(noise_level, damping=False):
    """
    데이터 로드 함수
    Args:
        noise_level: 사용할 noise level (예: 'clean', 'noise_40%', ...)
        damping: True이면 감쇠 데이터, False이면 비감쇠 데이터 사용
    Returns:
        time_train, time_test, time_valid, train_data, test_data, valid_data
    """
    if damping:
        file_path = f"data/single_pendulum_damped/{noise_level}/"
    else:
        file_path = f"data/single_pendulum_undamped/{noise_level}/"
    
    time_train = np.loadtxt(f'{file_path}{noise_level}_input_train.txt', delimiter='\t', skiprows=1)
    time_test  = np.loadtxt(f'{file_path}{noise_level}_input_test.txt', delimiter='\t', skiprows=1)
    time_valid = np.loadtxt(f'{file_path}{noise_level}_input_valid.txt', delimiter='\t', skiprows=1)
    train_data = np.loadtxt(f'{file_path}{noise_level}_output_train.txt', delimiter='\t', skiprows=1)
    test_data  = np.loadtxt(f'{file_path}{noise_level}_output_test.txt', delimiter='\t', skiprows=1)
    valid_data = np.loadtxt(f'{file_path}{noise_level}_output_valid.txt', delimiter='\t', skiprows=1)
    
    return time_train, time_test, time_valid, train_data, test_data, valid_data

def preprocess_data(time_data, data):
    """
    시간 데이터와 나머지 데이터를 결합하여 시간순으로 정렬한 후,
    출력 데이터는 오직 각가속도(angular acceleration)만 선택합니다.
    (원본 출력 데이터가 [angle, angular velocity, angular acceleration] 순서라고 가정)
    """
    combined = np.concatenate((time_data.reshape(-1, 1), data), axis=1)
    sorted_data = combined[np.argsort(combined[:, 0])]
    # 시간과 오직 마지막 열(각가속도)만 반환
    return sorted_data[:, 0], sorted_data[:, -1:]

# ============================================================
# PART 2: Model Definition (LSTM)
# ============================================================
class LSTMPendulum(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        """
        Args:
            input_size: 입력 차원 (여기서는 데이터 차원)
            hidden_size: LSTM 은닉층 크기
            output_size: 출력 차원 (여기서는 각가속도만 예측하므로 1)
            num_layers: LSTM 레이어 수
        """
        super(LSTMPendulum, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# ============================================================
# PART 3: Sequence Generation and Tensor Conversion
# ============================================================
def create_sequences(data, seq_length):
    """
    데이터를 시퀀스로 변환합니다.
    Args:
        data: (n, feature) 배열
        seq_length: 시퀀스 길이
    Returns:
        sequences, targets (numpy 배열)
    """
    sequences = [data[i:i+seq_length] for i in range(len(data) - seq_length)]
    targets = [data[i+seq_length] for i in range(len(data) - seq_length)]
    return np.array(sequences), np.array(targets)

def convert_to_tensor(X, y):
    """
    numpy 배열을 torch Tensor로 변환합니다.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

# ============================================================
# PART 4: Training & Validation Loop
# ============================================================
def train_model(model, criterion, optimizer, scheduler,
                X_train_tensor, y_train_tensor, X_valid_tensor, y_valid_tensor,
                num_epochs, batch_size, noise_level):
    train_losses, valid_losses = [], []
    num_batches = int(np.ceil(len(X_train_tensor) / batch_size))
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
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
        
        # Best model 저장
        if valid_loss.item() < best_valid_loss:
            best_valid_loss = valid_loss.item()
            best_model_path = f'example/trained_models/single_pen_LSTM_model_{noise_level}.pth'
            torch.save(model.state_dict(), best_model_path)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {valid_loss.item():.4f}')
    
    return train_losses, valid_losses

# ============================================================
# PART 5: Testing and Plotting
# ============================================================
def plot_results(time_test: np.ndarray, test_data: np.ndarray, predictions: np.ndarray, 
                 seq_length: int, noise_level: str, save_path: str):
    """
    결과 시각화 및 저장
    test_data와 predictions는 (n,1) 형태로 되어 있으며,
    여기서는 오직 각가속도만 플롯합니다.
    """
    plt.figure(figsize=(15, 5))
    
    plt.plot(time_test[seq_length:], test_data[seq_length:, 0], label='True Angular Acceleration', color='black', linestyle='-')
    plt.plot(time_test[seq_length:], predictions[:, 0], label='Predicted Angular Acceleration', color='red', linestyle='--')
    
    # 중앙 시간 기준
    extrapolation_start_time = time_test[len(time_test) // 2]
    plt.axvline(x=extrapolation_start_time, color='gray',linestyle='--', linewidth=2, label='Extrapolation Start')
    
    plt.title('Test Data Prediction (Angular Acceleration Only)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Acceleration (rad/s²)')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# ============================================================
# PART 6: Main Function
# ============================================================
def main():
    # 하이퍼파라미터 설정
    noise_level = 'clean'  # 예: 'clean', 'noise_40%', 등
    damping = True         # 감쇠 여부
    seq_length = 20
    hidden_size = 50
    output_size = 1        # 오직 각가속도만 예측하므로 출력 차원은 1
    num_layers = 2
    num_epochs = 200
    learning_rate = 0.001
    batch_size = 32

    # 데이터 로드 및 전처리
    time_train, time_test, time_valid, train_data, test_data, valid_data = data_loading(noise_level, damping)
    time_train_sorted, data_train_sorted = preprocess_data(time_train, train_data)
    time_test_sorted, data_test_sorted   = preprocess_data(time_test, test_data)
    time_valid_sorted, data_valid_sorted = preprocess_data(time_valid, valid_data)
    
    # 입력 차원 결정
    input_size = 1
    
    # 시퀀스 생성
    X_train, y_train = create_sequences(data_train_sorted, seq_length)
    X_valid, y_valid = create_sequences(data_valid_sorted, seq_length)
    X_test, y_test   = create_sequences(data_test_sorted, seq_length)
    
    # Tensor 변환
    X_train_tensor, y_train_tensor = convert_to_tensor(X_train, y_train)
    X_valid_tensor, y_valid_tensor = convert_to_tensor(X_valid, y_valid)
    X_test_tensor, y_test_tensor   = convert_to_tensor(X_test, y_test)
    
    # 모델 초기화
    model = LSTMPendulum(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    # 학습
    train_losses, valid_losses = train_model(model, criterion, optimizer, scheduler,
                                              X_train_tensor, y_train_tensor, 
                                              X_valid_tensor, y_valid_tensor,
                                              num_epochs, batch_size, noise_level)
    
    # 테스트 및 예측
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
    
    # 결과 플롯 및 저장
    results_fig_path = f'example/figs/single_pen_LSTM_fig_{noise_level}.png'
    plot_results(time_test_sorted, data_test_sorted, predictions, seq_length, noise_level, results_fig_path)
    print(f"Result figure saved to {results_fig_path}")
    
    mse = np.mean((data_test_sorted[seq_length:] - predictions) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")

if __name__ == "__main__":
    main()
