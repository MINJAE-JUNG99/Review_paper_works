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

# 데이터 로드 clean , noise_10% ...
noise_level = 'clean'  # 사용할 noise_level 설정
damping = False  # 댐핑 여부 설정

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

# # 플롯ting
# plt.figure(figsize=(15, 15))

# # 훈련 데이터 플롯
# plt.subplot(3, 1, 1)
# plt.plot(train_sorted[:, 0], train_sorted[:, 1], label='Train Angle', color='b')  # 각도 플롯
# plt.plot(train_sorted[:, 0], train_sorted[:, 2], label='Train Angular Acceleration', color='r')  # 각가속도 플롯
# plt.title('Training Data')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.legend()
# plt.grid()

# # 테스트 데이터 플롯
# plt.subplot(3, 1, 2)
# plt.plot(test_sorted[:, 0], test_sorted[:, 1], label='Test Angle', color='b')  # 각도 플롯
# plt.plot(test_sorted[:, 0], test_sorted[:, 2], label='Test Angular Acceleration', color='r')  # 각가속도 플롯
# plt.title('Test Data')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.legend()
# plt.grid()
# 테스트 데이터 플롯

# plt.subplot(3, 1, 2)
# plt.scatter(test_sorted[:, 0], test_sorted[:, 1], label='Test Angle', color='b', s=10, alpha=0.5)  # 각도 산점도
# plt.scatter(test_sorted[:, 0], test_sorted[:, 2], label='Test Angular Acceleration', color='r', s=10, alpha=0.5)  # 각가속도 산점도
# plt.title('Test Data')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.legend()
# plt.grid()


# # 검증 데이터 플롯
# plt.subplot(3, 1, 3)
# plt.plot(valid_sorted[:, 0], valid_sorted[:, 1], label='Valid Angle', color='b')  # 각도 플롯
# plt.plot(valid_sorted[:, 0], valid_sorted[:, 2], label='Valid Angular Acceleration', color='r')  # 각가속도 플롯
# plt.title('Validation Data')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.legend()
# plt.grid()

# plt.tight_layout()  # 서브플롯 간격 조정
# plt.savefig(f'LSTM/Dataset_ground_{noise_level}.png',dpi=200)  # 플롯 출력
# plt.show()

# 시간과 데이터를 분리
time_train_sorted = train_sorted[:, 0]  # 첫 번째 열: 시간
data_train_sorted = train_sorted[:, 1:]  # 나머지 열: 데이터 (각도 및 각속도 각가속도)

time_test_sorted = test_sorted[:, 0]  # 첫 번째 열: 시간
data_test_sorted = test_sorted[:, 1:]  # 나머지 열: 데이터

time_valid_sorted = valid_sorted[:, 0]  # 첫 번째 열: 시간
data_valid_sorted = valid_sorted[:, 1:]  # 나머지 열: 데이터


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
input_size = data_train_sorted.shape[1]
hidden_size = 50
output_size = 3
num_layers = 2
num_epochs = 200  # 에포크 수 증가
learning_rate = 0.001
batch_size = 32

# 데이터 전처리: LSTM 입력 형태로 변환
# 각 데이터 배열을 (samples, features) 형태로 변환
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]  # 다음 스텝 예측
        sequences.append(seq)
        targets.append(label)
    
    return np.array(sequences), np.array(targets)

seq_length = 20
X_train, y_train = create_sequences(data_train_sorted, seq_length)
X_valid, y_valid = create_sequences(data_valid_sorted, seq_length)
X_test, y_test = create_sequences(data_test_sorted, seq_length)

# Tensor로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 모델 초기화 및 손실 함수, 최적화 알고리즘 설정
model = LSTMPendulum(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 및 검증 루프
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    model.train()  # 모델을 훈련 모드로 설정
    optimizer.zero_grad()  # 기울기 초기화
    
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

# 테스트 데이터로 평가
X_test, y_test = create_sequences(data_test_sorted, seq_length)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    # 초기 시퀀스를 초기값으로 패딩
    initial_state = torch.tensor(data_test_sorted[0], dtype=torch.float32)
    padded_input = initial_state.repeat(seq_length, 1).unsqueeze(0)  # [1, seq_length, input_size]

    predictions = []
    for i in range(len(time_test_sorted)):
        output = model(padded_input)
        predictions.append(output.squeeze().numpy())

        # 다음 스텝을 위해 입력 시퀀스 업데이트
        if i < len(time_test_sorted) - 1:
            # 테스트 데이터에서 다음 값을 가져와 업데이트
            next_input = torch.tensor(data_test_sorted[i+1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            # 마지막 스텝에서는 모델의 출력을 사용
            next_input = output.unsqueeze(1)
        
        padded_input = torch.cat([padded_input[:, 1:, :], next_input], dim=1)

predictions = np.array(predictions)
# # 학습 및 검증 손실 그래프
# plt.subplot(3, 1, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(valid_losses, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# 테스트 데이터 예측 결과 - 각도
plt.subplot(3, 1, 2)
plt.plot(time_test_sorted, data_test_sorted[:, 0], label='True Angle', color='blue')
plt.plot(time_test_sorted, predictions[:, 0], label='Predicted Angle', color='red', linestyle='--')
plt.title('Test Data Prediction - Angle')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.legend()
plt.grid(True)

# 테스트 데이터 예측 결과 - 각가속도
plt.subplot(3, 1, 3)
plt.plot(time_test_sorted, data_test_sorted[:, 1], label='True Angular Acceleration', color='green')
plt.plot(time_test_sorted, predictions[:, 1], label='Predicted Angular Acceleration', color='orange', linestyle='--')
plt.title('Test Data Prediction - Angular Acceleration')
plt.xlabel('Time')
plt.ylabel('Angular Acceleration')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'LSTM/LSTM_results_full_{noise_level}.png', dpi=200)
plt.show()

# # 결과 플롯
# plt.figure(figsize=(15, 10))

# # 학습 및 검증 손실 그래프
# plt.subplot(2, 1, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(valid_losses, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()

# # 테스트 데이터 예측 결과
# plt.subplot(2, 1, 2)
# plt.scatter(time_test_sorted, data_test_sorted[:, 0], label='True Angle', color='blue', s=10, alpha=0.5)
# plt.scatter(time_test_sorted, predictions[:, 0], label='Predicted Angle', color='red', s=10, alpha=0.5)
# plt.title('Test Data Prediction (Full Time Series)')
# plt.xlabel('Time')
# plt.ylabel('Angle')
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.savefig(f'LSTM/LSTM_results_full_{noise_level}.png', dpi=200)
# plt.show()

# MSE 계산
mse = np.mean((data_test_sorted - predictions)**2)
print(f'Mean Squared Error: {mse:.4f}')

# 결과를 CSV 파일로 저장
#results = np.column_stack((time_test_sorted, data_test_sorted, predictions))
#np.savetxt(f'LSTM/LSTM_predictions_{noise_level}.csv', results, delimiter=',', 
#          header='Time,True_Angle,True_Angular_Acceleration,Predicted_Angle,Predicted_Angular_Acceleration', 
#           comments='')