import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# data_loading 함수 정의
def data_loading(noise_level, damping=False):
    if damping:
        file_path = f"data/test/single_pendulum_damped/{noise_level}/"
    else:
        file_path = f"data/test/single_pendulum_undamped/{noise_level}/"

    # 탭 구분자를 사용하여 데이터를 읽어옴
    time_train = np.loadtxt(f'{file_path}{noise_level}_input_train.txt', delimiter='\t', skiprows=1)
    time_test = np.loadtxt(f'{file_path}{noise_level}_input_test.txt', delimiter='\t', skiprows=1)
    time_valid = np.loadtxt(f'{file_path}{noise_level}_input_valid.txt', delimiter='\t', skiprows=1)
    train_data = np.loadtxt(f'{file_path}{noise_level}_output_train.txt', delimiter='\t', skiprows=1)
    test_data = np.loadtxt(f'{file_path}{noise_level}_output_test.txt', delimiter='\t', skiprows=1)
    valid_data = np.loadtxt(f'{file_path}{noise_level}_output_valid.txt', delimiter='\t', skiprows=1)

    return time_train, time_test, time_valid, train_data, test_data, valid_data

# 데이터 로드
noise_level = 'clean'  # 사용할 noise_level 설정
damping = True  # 댐핑 여부 설정

time_train, time_test, time_valid, train_data, test_data, valid_data = data_loading(noise_level, damping)

# 1차원 배열을 2차원 배열로 변환
time_train = time_train.reshape(-1, 1)  # (n_samples, 1) 형태로 변환
time_test = time_test.reshape(-1, 1)    # (n_samples, 1) 형태로 변환
time_valid = time_valid.reshape(-1, 1)  # (n_samples, 1) 형태로 변환

# 시간과 데이터 결합
train_combined = np.concatenate((time_train, train_data), axis=1)
test_combined = np.concatenate((time_test, test_data), axis=1)
valid_combined = np.concatenate((time_valid, valid_data), axis=1)

# 시간 순서대로 정렬
train_sorted = train_combined[np.argsort(train_combined[:, 0])]
test_sorted = test_combined[np.argsort(test_combined[:, 0])]
valid_sorted = valid_combined[np.argsort(valid_combined[:, 0])]

# 시간과 데이터를 분리
time_train_sorted = train_sorted[:, 0]  # 첫 번째 열: 시간
data_train_sorted = train_sorted[:, 1:]  # 나머지 열: 데이터 (각도 및 각속도 각가속도)

time_test_sorted = test_sorted[:, 0]  # 첫 번째 열: 시간
data_test_sorted = test_sorted[:, 1:]  # 나머지 열: 데이터

time_valid_sorted = valid_sorted[:, 0]  # 첫 번째 열: 시간
data_valid_sorted = valid_sorted[:, 1:]  # 나머지 열: 데이터

# MLP 모델 정의
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 하이퍼파라미터 설정
input_size = data_train_sorted.shape[1]  # 각도 및 각속도 각가속도
hidden_size = 50
output_size = 3  # 예측할 출력 크기 (각도 및 각속도 각가속도)
num_epochs = 200  # 에포크 수 증가
learning_rate = 0.001
batch_size = 32

# 데이터 전처리: LSTM 입력 형태로 변환
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]  # 다음 스텝 예측
        sequences.append(seq)
        targets.append(label)
    
    return np.array(sequences), np.array(targets)

# 시퀀스 길이 설정
seq_length = 20
X_train, y_train = create_sequences(data_train_sorted, seq_length)
X_valid, y_valid = create_sequences(data_valid_sorted, seq_length)
X_test, y_test = create_sequences(data_test_sorted, seq_length)

# Tensor로 변환
X_train_tensor = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32)  # (samples, features)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid.reshape(X_valid.shape[0], -1), dtype=torch.float32)  # (samples, features)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32)  # (samples, features)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 모델 초기화 및 손실 함수, 최적화 알고리즘 설정
model = MLPModel(input_size=input_size * seq_length, hidden_size=hidden_size, output_size=output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 및 검증 루프
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    model.train()  # 모델을 훈련 모드로 설정
    optimizer.zero_grad()  # 기울기 초기화
    
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

# 테스트 데이터로 평가
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)

# 결과 플롯
plt.figure(figsize=(15, 5))

# 각도
plt.plot(time_test_sorted[seq_length:], data_test_sorted[seq_length:, 0], label='Reference Angle', color='black', linestyle='-')
plt.plot(time_test_sorted[seq_length:], y_test_pred.numpy()[:, 0], label='Predicted Angle', color='red', linestyle='--')

# 각속도
plt.plot(time_test_sorted[seq_length:], data_test_sorted[seq_length:, 1], label='Reference Angular Velocity', color='black', linestyle='-')
plt.plot(time_test_sorted[seq_length:], y_test_pred.numpy()[:, 1], label='Predicted Angular Velocity', color='blue', linestyle='--')

# 각가속도
plt.plot(time_test_sorted[seq_length:], data_test_sorted[seq_length:, 2], label='Reference Angular Acceleration', color='black', linestyle='-')
plt.plot(time_test_sorted[seq_length:], y_test_pred.numpy()[:, 2], label='Predicted Angular Acceleration', color='green', linestyle='--')


# Extrapolation 구간 시작 시간 설정 (중앙 5초 구간)
extrapolation_start_time = time_test[len(time_test) // 2]  # 중앙 시간값
# Extrapolation 구간 시작점에 수직선 추가
plt.axvline(x=extrapolation_start_time, color='gray', linestyle='--', linewidth=2, 
                label='Extrapolation Start')

plt.title('Test Data Prediction (Angle, Angular Velocity, Angular Acceleration)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'MLP/MLP_results_seq_{noise_level}.png', dpi=200)
plt.show()


# MSE 계산
mse = np.mean((data_test_sorted[seq_length:] - y_test_pred.numpy())**2)
print(f'Mean Squared Error: {mse:.4f}')

# 결과를 CSV 파일로 저장
results = np.column_stack((time_test_sorted[seq_length:], data_test_sorted[seq_length:], y_test_pred.numpy()))
np.savetxt(f'MLP/MLP_predictions_{noise_level}.csv', results, delimiter=',', 
           header='Time,True_Angle,True_Angular_Acceleration,Predicted_Angle,Predicted_Angular_Acceleration', 
           comments='')