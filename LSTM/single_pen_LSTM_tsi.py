import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 함수는 동일
def data_loading(noise_level, damping=False):
    file_path = f"data/test/single_pendulum_damped/{noise_level}/" if damping else f"data/test/single_pendulum_undamped/{noise_level}/"
    
    time_train = np.loadtxt(f'{file_path}{noise_level}_input_train.txt', delimiter='\t', skiprows=1)
    time_test = np.loadtxt(f'{file_path}{noise_level}_input_test.txt', delimiter='\t', skiprows=1)
    time_valid = np.loadtxt(f'{file_path}{noise_level}_input_valid.txt', delimiter='\t', skiprows=1)
    train_data = np.loadtxt(f'{file_path}{noise_level}_output_train.txt', delimiter='\t', skiprows=1)
    test_data = np.loadtxt(f'{file_path}{noise_level}_output_test.txt', delimiter='\t', skiprows=1)
    valid_data = np.loadtxt(f'{file_path}{noise_level}_output_valid.txt', delimiter='\t', skiprows=1)

    return time_train, time_test, time_valid, train_data, test_data, valid_data

# 데이터 로드
noise_level = 'clean'
damping = True
time_train, time_test, time_valid, train_data, test_data, valid_data = data_loading(noise_level, damping)

# 데이터 전처리 - 시간 데이터만 사용
def preprocess_data(time_data, target_data):
    combined = np.column_stack((time_data, target_data))
    sorted_data = combined[np.argsort(combined[:, 0])]
    return sorted_data[:, 0], sorted_data[:, 1:]

time_train_sorted, data_train_sorted = preprocess_data(time_train, train_data)
time_test_sorted, data_test_sorted = preprocess_data(time_test, test_data)
time_valid_sorted, data_valid_sorted = preprocess_data(time_valid, valid_data)

# 시퀀스 생성 함수 수정 - 입력은 시간만, 출력은 상태 변수
def create_sequences(time_data, target_data, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(time_data) - seq_length):
        # 시간 시퀀스를 입력으로 사용
        seq = time_data[i:i + seq_length]
        # 다음 시점의 상태 변수를 타겟으로 사용
        target = target_data[i + seq_length]
        
        sequences.append(seq.reshape(-1, 1))  # shape: (seq_length, 1)
        targets.append(target)
        
    return np.array(sequences), np.array(targets)

# LSTM 모델 정의 - 입력 크기를 1로 변경 (시간만 입력)
class LSTMPendulum(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=2):
        super(LSTMPendulum, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# 하이퍼파라미터 설정
hidden_size = 50
output_size = 3  # 각도, 각속도, 각가속도
num_layers = 2
num_epochs = 2000
learning_rate = 0.001
batch_size = 32
seq_length = 20

# 데이터를 시퀀스로 변환
X_train, y_train = create_sequences(time_train_sorted, data_train_sorted, seq_length)
X_valid, y_valid = create_sequences(time_valid_sorted, data_valid_sorted, seq_length)
X_test, y_test = create_sequences(time_test_sorted, data_test_sorted, seq_length)

# Tensor로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 모델 초기화 및 학습 설정
model = LSTMPendulum(hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 및 검증 루프
train_losses, valid_losses = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / (len(X_train_tensor) / batch_size)
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        y_valid_pred = model(X_valid_tensor)
        valid_loss = criterion(y_valid_pred, y_valid_tensor)
    valid_losses.append(valid_loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {valid_loss.item():.4f}')

# 테스트 데이터로 예측
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# 결과 플롯 (손실 그래프 제거)
plt.figure(figsize=(15, 5))

# 색상 및 레이블 설정
colors = ['red', 'blue', 'green']
labels = ['Angle', 'Angular Velocity', 'Angular Acceleration']

# Extrapolation 구간 시작 시간 설정 (중앙 5초 구간)
extrapolation_start_time = time_test_sorted[len(time_test_sorted) // 2]  # 중앙 시간값

for i in range(3):
    plt.plot(time_test_sorted[seq_length:], data_test_sorted[seq_length:, i], 
             label='Reference', color='black', linestyle='-')
    plt.plot(time_test_sorted[seq_length:], predictions[:, i], 
             label=f'Predicted {labels[i]}', color=colors[i], linestyle='--')

# Extrapolation 구간 시작점에 수직선 추가
plt.axvline(x=extrapolation_start_time, color='red', linewidth=2, label='Extrapolation Start')

plt.title('Test Data Prediction')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(f'LSTM/LSTM_results_time_only_sequence_{noise_level}.png', dpi=200)
plt.show()


# 결과를 CSV 파일로 저장
results = np.column_stack((time_test_sorted[seq_length:], data_test_sorted[seq_length:], predictions))
np.savetxt(f'LSTM/LSTM_predictions_time_only_{noise_level}.csv', results, delimiter=',', 
           header='Time,True_Angle,True_Angular_Velocity,True_Angular_Acceleration,Predicted_Angle,Predicted_Angular_Velocity,Predicted_Angular_Acceleration', 
           comments='')