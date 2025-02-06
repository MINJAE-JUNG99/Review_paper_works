import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 함수
def data_loading(noise_level, damping=False):
    file_path = f"data/test/single_pendulum_damped/{noise_level}/" if damping else f"data/test/single_pendulum_undamped/{noise_level}/"
    
    # 데이터 로드
    time_train = np.loadtxt(f'{file_path}{noise_level}_input_train.txt', delimiter='\t', skiprows=1)[:, 0]
    time_test = np.loadtxt(f'{file_path}{noise_level}_input_test.txt', delimiter='\t', skiprows=1)[:, 0]
    time_valid = np.loadtxt(f'{file_path}{noise_level}_input_valid.txt', delimiter='\t', skiprows=1)[:, 0]
    train_data = np.loadtxt(f'{file_path}{noise_level}_output_train.txt', delimiter='\t', skiprows=1)
    test_data = np.loadtxt(f'{file_path}{noise_level}_output_test.txt', delimiter='\t', skiprows=1)
    valid_data = np.loadtxt(f'{file_path}{noise_level}_output_valid.txt', delimiter='\t', skiprows=1)

    return time_train, time_test, time_valid, train_data, test_data, valid_data

# 데이터 로드
noise_level = 'clean'
damping = True
time_train, time_test, time_valid, train_data, test_data, valid_data = data_loading(noise_level, damping)

# 시퀀스 생성 함수
def create_sequences(data, seq_length):
    sequences = [data[i:i + seq_length] for i in range(len(data) - seq_length)]
    targets = [data[i + seq_length] for i in range(len(data) - seq_length)]
    return np.array(sequences).reshape(-1, seq_length, 1), np.array(targets)

# 하이퍼파라미터 설정
hidden_size = 50
output_size = train_data.shape[1]
num_layers = 2
num_epochs = 200
learning_rate = 0.001
batch_size = 32
seq_length = 20

# 데이터를 시퀀스로 변환
X_train, y_train = create_sequences(time_train, seq_length)
X_valid, y_valid = create_sequences(time_valid, seq_length)
X_test, y_test = create_sequences(time_test, seq_length)

# Tensor로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# LSTM 모델 정의
class LSTMPendulum(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMPendulum, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# 모델 초기화 및 손실 함수, 최적화 알고리즘 설정
model = LSTMPendulum(input_size=1, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
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

# 결과 저장
results = np.column_stack((time_test[seq_length:], y_test, predictions))
np.savetxt(f'LSTM/LSTM_predictions_{noise_level}.csv', results, delimiter=',', 
           header='Time,True_Angle,True_Angular_Velocity,True_Angular_Acceleration,Predicted_Angle,Predicted_Angular_Velocity,Predicted_Angular_Acceleration', 
           comments='')
