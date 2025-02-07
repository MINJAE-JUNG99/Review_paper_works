import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 함수
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

# 데이터 전처리 - 시간과 상태 변수를 모두 포함
def preprocess_data(time_data, state_data):
    # 시간과 상태 변수를 결합
    combined = np.column_stack((time_data, state_data))
    sorted_indices = np.argsort(combined[:, 0])
    sorted_data = combined[sorted_indices]
    
    return sorted_data[:, 0], sorted_data[:, 1:]  # 시간과 상태 변수 분리 반환

time_train_sorted, data_train_sorted = preprocess_data(time_train, train_data)
time_test_sorted, data_test_sorted = preprocess_data(time_test, test_data)
time_valid_sorted, data_valid_sorted = preprocess_data(time_valid, valid_data)

# 시퀀스 생성 함수 수정 - 시간과 상태 변수를 모두 포함
def create_sequences(time_data, state_data, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(time_data) - seq_length):
        # 시간과 상태 변수를 결합하여 시퀀스 생성
        time_seq = time_data[i:i + seq_length].reshape(-1, 1)
        state_seq = state_data[i:i + seq_length]
        combined_seq = np.hstack((time_seq, state_seq))
        
        sequences.append(combined_seq)
        targets.append(state_data[i + seq_length])
    
    return np.array(sequences), np.array(targets)

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

# 하이퍼파라미터 설정
input_size = 4  # 시간(1) + 상태변수(3)
hidden_size = 50
output_size = 3  # 상태변수(3)
num_layers = 2
num_epochs = 200
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
model = LSTMPendulum(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
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

# 테스트 데이터로 예측 (다음 스텝을 예측하며 진행)
model.eval()
with torch.no_grad():
    predictions = []
    current_sequence = X_test_tensor[0]  # 초기 시퀀스

    # 첫 번째 시퀀스의 마지막 시간값 추출
    last_time = current_sequence[-1, 0].item()
    
    # 시간 간격 계산 (마지막 두 시점 간의 차이)
    time_step = current_sequence[-1, 0].item() - current_sequence[-2, 0].item()
    
    # 전체 예측 구간에 대해 반복
    for i in range(len(X_test_tensor)):
        # 현재 시퀀스로 다음 상태 예측
        current_input = current_sequence.unsqueeze(0)  # 배치 차원 추가
        next_state = model(current_input)
        predictions.append(next_state.numpy()[0])
        
        # 다음 시퀀스 준비
        # 1. 새로운 시간값 계산
        last_time += time_step
        
        # 2. 예측된 상태와 새로운 시간값 결합
        new_step = torch.tensor([last_time] + next_state[0].tolist(), dtype=torch.float32)
        
        # 3. 시퀀스 업데이트 (가장 오래된 스텝 제거하고 새로운 스텝 추가)
        current_sequence = torch.cat([current_sequence[1:], new_step.unsqueeze(0)])

    predictions = np.array(predictions)

# 결과 플롯
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
plt.savefig(f'LSTM/LSTM_results_time_plus_predstep_{noise_level}.png', dpi=200)
plt.show()


# MSE 계산
mse = np.mean((data_test_sorted[seq_length:] - predictions) ** 2)
print(f'Mean Squared Error: {mse:.4f}')

# 결과를 CSV 파일로 저장
results = np.column_stack((time_test_sorted[seq_length:], data_test_sorted[seq_length:], predictions))
np.savetxt(f'LSTM/LSTM_predictions_with_time_{noise_level}.csv', results, delimiter=',', 
           header='Time,True_Angle,True_Angular_Velocity,True_Angular_Acceleration,Predicted_Angle,Predicted_Angular_Velocity,Predicted_Angular_Acceleration', 
           comments='')