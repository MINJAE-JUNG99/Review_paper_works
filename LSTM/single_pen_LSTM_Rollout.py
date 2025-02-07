import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

# 데이터 로드 함수
def data_loading(noise_level, damping=False):
    file_path = f"data/test/single_pendulum_damped/{noise_level}/" if damping else f"data/test/single_pendulum_undamped/{noise_level}/"
    
    train_data = np.loadtxt(f'{file_path}{noise_level}_output_train.txt', delimiter='\t', skiprows=1)
    test_data = np.loadtxt(f'{file_path}{noise_level}_output_test.txt', delimiter='\t', skiprows=1)
    valid_data = np.loadtxt(f'{file_path}{noise_level}_output_valid.txt', delimiter='\t', skiprows=1)

    return train_data, test_data, valid_data

# 데이터 로드
noise_level = 'clean'
damping = True
train_data, test_data, valid_data = data_loading(noise_level, damping)

# 시퀀스 생성 함수
def create_sequences(state_data, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(state_data) - seq_length):
        sequences.append(state_data[i:i + seq_length])
        targets.append(state_data[i + seq_length])
    
    return np.array(sequences), np.array(targets)

# LSTM 모델 정의 (Dropout 추가)
class LSTMPendulum(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(LSTMPendulum, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# 하이퍼파라미터 설정
input_size = 3  # 상태변수(각도, 각속도, 각가속도)
hidden_size = 100  # Hidden Size 증가
output_size = 3
num_layers = 3  # LSTM 층 수 증가
num_epochs = 200
learning_rate = 0.001
batch_size = 32
seq_length = 30  # 시퀀스 길이 증가
dropout = 0.2  # Dropout 추가

# 데이터를 시퀀스로 변환
X_train, y_train = create_sequences(train_data, seq_length)
X_valid, y_valid = create_sequences(valid_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Tensor로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 모델 초기화 및 학습 설정
model = LSTMPendulum(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # 학습률 스케줄링

# 학습 루프
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
    
    scheduler.step()  # 학습률 업데이트
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {valid_loss.item():.4f}')

# Rollout 방식으로 예측
def rollout_predict(model, initial_sequence, num_steps):
    model.eval()
    predictions = []
    current_sequence = initial_sequence.clone()  # 초기 시퀀스 복사

    for _ in range(num_steps):
        with torch.no_grad():
            # 현재 시퀀스로 다음 상태 예측
            current_input = current_sequence.unsqueeze(0)  # 배치 차원 추가
            next_state = model(current_input)
            predictions.append(next_state.numpy()[0])
            
            # 다음 시퀀스 업데이트
            current_sequence = torch.cat([current_sequence[1:], next_state], dim=0)
    
    return np.array(predictions)

# Rollout 예측 수행
initial_sequence = X_test_tensor[0]  # 초기 시퀀스
num_steps = len(X_test_tensor)  # 테스트 데이터 길이만큼 예측
predictions = rollout_predict(model, initial_sequence, num_steps)

# 결과 플롯
plt.figure(figsize=(15, 5))
colors = ['red', 'blue', 'green']
labels = ['Angle', 'Angular Velocity', 'Angular Acceleration']

for i in range(3):
    plt.plot(y_test[:, i], label=f'Reference {labels[i]}', color='black', linestyle='-')
    plt.plot(predictions[:, i], label=f'Predicted {labels[i]}', color=colors[i], linestyle='--')

plt.title('Test Data Prediction (Rollout)')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.savefig(f'LSTM/LSTM_results_rollout_{noise_level}.png', dpi=200)
plt.show()

# MSE 계산
mse = np.mean((y_test - predictions) ** 2)
print(f'Mean Squared Error: {mse:.4f}')

# 결과를 CSV 파일로 저장
results = np.column_stack((y_test, predictions))
np.savetxt(f'LSTM/LSTM_predictions_rollout_{noise_level}.csv', results, delimiter=',', 
           header='True_Angle,True_Angular_Velocity,True_Angular_Acceleration,Predicted_Angle,Predicted_Angular_Velocity,Predicted_Angular_Acceleration', 
           comments='')