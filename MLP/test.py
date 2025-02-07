import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 함수
def data_loading(noise_level, damping=False):
    if damping:
        file_path = f"data/test/single_pendulum_damped/{noise_level}/"
    else:
        file_path = f"data/test/single_pendulum_undamped/{noise_level}/"

    time_train = np.loadtxt(f'{file_path}{noise_level}_input_train.txt', delimiter='\t', skiprows=1)
    time_test = np.loadtxt(f'{file_path}{noise_level}_input_test.txt', delimiter='\t', skiprows=1)
    time_valid = np.loadtxt(f'{file_path}{noise_level}_input_valid.txt', delimiter='\t', skiprows=1)
    train_data = np.loadtxt(f'{file_path}{noise_level}_output_train.txt', delimiter='\t', skiprows=1)
    test_data = np.loadtxt(f'{file_path}{noise_level}_output_test.txt', delimiter='\t', skiprows=1)
    valid_data = np.loadtxt(f'{file_path}{noise_level}_output_valid.txt', delimiter='\t', skiprows=1)
    
    return time_train, time_test, time_valid, train_data, test_data, valid_data

# 데이터 로드 및 정규화
noise_level = 'clean'
damping = True

time_train, time_test, time_valid, train_data, test_data, valid_data = data_loading(noise_level, damping)

# 시간 정렬
train_sorted = np.column_stack((time_train, train_data))[np.argsort(time_train)]
test_sorted = np.column_stack((time_test, test_data))[np.argsort(time_test)]
valid_sorted = np.column_stack((time_valid, valid_data))[np.argsort(time_valid)]

# 정렬된 데이터 분리
time_train_sorted, data_train_sorted = train_sorted[:, 0], train_sorted[:, 1:]
time_test_sorted, data_test_sorted = test_sorted[:, 0], test_sorted[:, 1:]
time_valid_sorted, data_valid_sorted = valid_sorted[:, 0], valid_sorted[:, 1:]

# 데이터 정규화
mean_train, std_train = data_train_sorted.mean(axis=0), data_train_sorted.std(axis=0)
data_train_norm = (data_train_sorted - mean_train) / std_train
data_test_norm = (data_test_sorted - mean_train) / std_train
data_valid_norm = (data_valid_sorted - mean_train) / std_train

# PyTorch 텐서 변환
def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

# 시간 제외하고 데이터만 학습에 사용
X_train_tensor = to_tensor(data_train_norm)  # 시간 제외한 데이터
y_train_tensor = to_tensor(data_train_norm)  # 데이터는 출력값과 동일 (이 예에서는 True 값 그대로 사용)
X_valid_tensor = to_tensor(data_valid_norm)  # 동일
y_valid_tensor = to_tensor(data_valid_norm)
X_test_tensor = to_tensor(data_test_norm)  # 동일
y_test_tensor = to_tensor(data_test_norm)

# MLP 모델 정의
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

# 하이퍼파라미터 설정
input_size = X_train_tensor.shape[1]  # 수정: 입력 차원 업데이트
hidden_size = 100
output_size = y_train_tensor.shape[1]
num_epochs = 200
learning_rate = 0.001

# 모델 초기화
model = MLPModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# 학습 및 검증 루프
train_losses, valid_losses = [], []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    scheduler.step()

    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        y_valid_pred = model(X_valid_tensor)
        valid_loss = criterion(y_valid_pred, y_valid_tensor)
        valid_losses.append(valid_loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.6f}, Validation Loss: {valid_loss.item():.6f}')

# 테스트 예측
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor).numpy()

# 정규화 복원
y_test_pred = (y_test_pred * std_train) + mean_train

# 결과 플롯 함수
def plot_results(time_test: np.ndarray, test_data: np.ndarray, predictions: np.ndarray, 
                 noise_level: str, save_path: str):
    """결과 시각화 및 저장 (수정된 버전)"""
    plt.figure(figsize=(15, 5))

    # 색상 및 레이블 설정
    colors = ['red', 'blue', 'green']
    labels = ['Angle', 'Angular Velocity', 'Angular Acceleration']

    # Extrapolation 구간 시작 시간 설정 (중앙 5초 구간)
    extrapolation_start_time = time_test[len(time_test) // 2]  # 중앙 시간값

    for i in range(3):
        plt.plot(time_test, test_data[:, i], label=f'True {labels[i]}', 
                 color='black', linestyle='-')
        plt.plot(time_test, predictions[:, i], label=f'Predicted {labels[i]}', 
                 color=colors[i], linestyle='--')

    # Extrapolation 구간 시작점에 수직선 추가
    plt.axvline(x=extrapolation_start_time, color='gray', linestyle='--', linewidth=2, 
                label='Extrapolation Start')

    plt.title('Test Data Prediction')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# 저장할 디렉토리 확인
os.makedirs("MLP", exist_ok=True)

# 결과 플롯 호출
save_path = f'MLP/MLP_results_{noise_level}.png'
plot_results(time_test_sorted, data_test_sorted, y_test_pred, noise_level, save_path)

# MSE 계산 및 CSV 저장
mse = np.mean((data_test_sorted - y_test_pred) ** 2)
print(f'Mean Squared Error: {mse:.6f}')

np.savetxt(f'MLP/MLP_predictions_test_{noise_level}.csv', np.column_stack((time_test_sorted, data_test_sorted, y_test_pred)), delimiter=',', header='Time,True_Angle,True_Angular_Velocity,True_Angular_Acceleration,Pred_Angle,Pred_Angular_Velocity,Pred_Angular_Acceleration', comments='')
