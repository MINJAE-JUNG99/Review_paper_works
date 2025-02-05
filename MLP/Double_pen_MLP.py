import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# data_loading 함수 정의
def data_loading(noise_level, chaotic=False):
    if chaotic:
        file_path = f"data/uniform/Double_pen_chaotic_uniform/{noise_level}/"
    else:
        file_path = f"data/uniform/Double_pen_moderate_uniform/{noise_level}/"
    if chaotic:
        # 탭 구분자를 사용하여 데이터를 읽어옴
        time_train = np.loadtxt(f'{file_path}{noise_level}_chaotic_input_train.txt', delimiter='\t', skiprows=1)
        time_test = np.loadtxt(f'{file_path}{noise_level}_chaotic_input_test.txt', delimiter='\t', skiprows=1)
        time_valid = np.loadtxt(f'{file_path}{noise_level}_chaotic_input_valid.txt', delimiter='\t', skiprows=1)
        train_data = np.loadtxt(f'{file_path}{noise_level}_chaotic_output_train.txt', delimiter='\t', skiprows=1)
        test_data = np.loadtxt(f'{file_path}{noise_level}_chaotic_output_test.txt', delimiter='\t', skiprows=1)
        valid_data = np.loadtxt(f'{file_path}{noise_level}_chaotic_output_valid.txt', delimiter='\t', skiprows=1)
    else:
        # 탭 구분자를 사용하여 데이터를 읽어옴
        time_train = np.loadtxt(f'{file_path}{noise_level}_moderate_input_train.txt', delimiter='\t', skiprows=1)
        time_test = np.loadtxt(f'{file_path}{noise_level}_moderate_input_test.txt', delimiter='\t', skiprows=1)
        time_valid = np.loadtxt(f'{file_path}{noise_level}_moderate_input_valid.txt', delimiter='\t', skiprows=1)
        train_data = np.loadtxt(f'{file_path}{noise_level}_moderate_output_train.txt', delimiter='\t', skiprows=1)
        test_data = np.loadtxt(f'{file_path}{noise_level}_moderate_output_test.txt', delimiter='\t', skiprows=1)
        valid_data = np.loadtxt(f'{file_path}{noise_level}_moderate_output_valid.txt', delimiter='\t', skiprows=1)    
    

    return time_train, time_test, time_valid, train_data, test_data, valid_data

# 데이터 로드
noise_level = 'clean'  # 사용할 noise_level 설정
chaotic = False  # 댐핑 여부 설정

time_train, time_test, time_valid, train_data, test_data, valid_data = data_loading(noise_level, chaotic)

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
input_size = data_train_sorted.shape[1]  # 각도 및 각속도 각가속도 (2개의 진자)
hidden_size = 50
output_size = 4  # 예측할 출력 크기 (각도 및 각속도 각가속도) (2개의 진자)
num_epochs = 200  # 에포크 수 증가
learning_rate = 0.001
batch_size = 32

def create_sequences_with_initial_padding(data, seq_length):
    # 첫 번째 값으로 패딩 추가
    padded_data = np.pad(data, ((seq_length, 0), (0, 0)), mode='constant', constant_values=data[0])
    sequences = []
    targets = []

    for i in range(len(data)):
        seq = padded_data[i:i + seq_length]
        if i + seq_length < len(data):
            label = data[i + seq_length]  # 다음 스텝 예측
        else:
            label = data[-1]  # 마지막 값으로 예측 (패딩 이후 마지막 값으로 처리)
        sequences.append(seq)
        targets.append(label)
    
    return np.array(sequences), np.array(targets)

# 시퀀스 생성
seq_length = 20
X_train, y_train = create_sequences_with_initial_padding(data_train_sorted, seq_length)
X_valid, y_valid = create_sequences_with_initial_padding(data_valid_sorted, seq_length)
X_test, y_test = create_sequences_with_initial_padding(data_test_sorted, seq_length)

# 텐서로 변환
X_train_tensor = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid.reshape(X_valid.shape[0], -1), dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 모델 초기화 및 손실 함수, 최적화 알고리즘 설정
model = MLPModel(input_size=input_size * seq_length, hidden_size=hidden_size, output_size=output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 최적의 모델 저장을 위한 변수 초기화
best_valid_loss = float('inf')  # 초기값을 무한대로 설정
best_model_path = 'MLP/model/Doublepen/best_model_mlp.pt'  # 모델 저장 경로

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

    # 검증 손실이 가장 낮은 모델 저장
    if valid_loss.item() < best_valid_loss:
        best_valid_loss = valid_loss.item()
        torch.save(model.state_dict(), best_model_path)  # 모델 가중치 저장
        print(f"Epoch {epoch + 1}: Validation loss improved to {best_valid_loss:.4f}. Model saved.")
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {valid_loss.item():.4f}')

print(f"Training complete. Best validation loss: {best_valid_loss:.4f}. Model saved to '{best_model_path}'.")

# 저장된 모델 로드
model = MLPModel(input_size=input_size * seq_length, hidden_size=hidden_size, output_size=output_size)
model.load_state_dict(torch.load(best_model_path))  
model.eval()  # 평가 모드로 설정

# 테스트 데이터로 평가
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_test_pred = y_test_pred.numpy()  # 예측 결과를 NumPy 배열로 변환
    y_test_actual = y_test_tensor.numpy()  # 실제 값을 NumPy 배열로 변환

output_path = "visualize/MLP/Doublepen_predictions.npy"

# NumPy 배열로 저장
np.save(output_path, y_test_pred)

print(f"Prediction data saved to {output_path}")

# 플롯 그리기
time_axis = np.arange(seq_length, seq_length + len(y_test_actual))  # 시간 축 생성

# 각 진자의 데이터 추출
# 첫 번째 진자의 각도, 각속도, 각가속도
theta1_actual = y_test_actual[:, 0]
theta1_pred = y_test_pred[:, 0]


# 두 번째 진자의 각도, 각속도, 각가속도
theta2_actual = y_test_actual[:, 2]
theta2_pred = y_test_pred[:, 2]


# 플롯 설정
plt.figure(figsize=(15, 10))

# 첫 번째 진자
plt.subplot(3, 2, 1)
plt.plot(time_axis, theta1_actual, label='Actual')
plt.plot(time_axis, theta1_pred, label='Predicted', linestyle='--')
plt.title('Pendulum 1: Angle (Theta1)')
plt.xlabel('Time')
plt.ylabel('Angle (rad)')
plt.legend()

# 두 번째 진자
plt.subplot(3, 2, 2)
plt.plot(time_axis, theta2_actual, label='Actual')
plt.plot(time_axis, theta2_pred, label='Predicted', linestyle='--')
plt.title('Pendulum 2: Angle (Theta2)')
plt.xlabel('Time')
plt.ylabel('Angle (rad)')
plt.legend()

plt.show()