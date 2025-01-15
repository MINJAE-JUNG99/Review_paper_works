import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# LSTM 모델 정의
class LSTMPendulum(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPendulum, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 마지막 타임스텝의 출력을 사용
        return output

# 댐핑된 단순 진자 시스템 클래스
class DampedSinglePendulumSystem(nn.Module):
    def __init__(self, length, mass, gravity, damping):
        super(DampedSinglePendulumSystem, self).__init__()
        self.length = length
        self.mass = mass
        self.gravity = gravity
        self.damping = damping  # 감쇠 계수 추가

    def forward(self, t, y):
        """
        Calculates the derivative dy/dt for the damped single pendulum.
        """
        theta, omega = y[:, 0:1], y[:, 1:2]
        # 감쇠가 있는 단일 진자의 운동 방정식 계산
        dy_dt = torch.zeros_like(y)
        dy_dt[:, 0] = omega  # d(theta)/dt = omega
        dy_dt[:, 1] = -(self.gravity / self.length) * torch.sin(theta) - (self.damping / (self.mass * self.length**2)) * omega  # 감쇠항 추가
        return dy_dt

# RK4를 사용한 노이즈가 추가된 데이터 생성 함수 (PyTorch 사용)
def generate_noisy_data_rk4(system, initial_state, t, noise_level):
    num_points = len(t)
    solution = torch.zeros((num_points, 2))
    solution[0] = initial_state

    for i in range(1, num_points):
        h = t[i] - t[i - 1]
        y = solution[i - 1].unsqueeze(0)

        k1 = system(t[i - 1], y)
        k2 = system(t[i - 1] + h / 2, y + h * k1 / 2)
        k3 = system(t[i - 1] + h / 2, y + h * k2 / 2)
        k4 = system(t[i], y + h * k3)

        solution[i] = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # 가우시안 노이즈 추가
    max_amplitude = torch.max(torch.abs(solution))
    noise = torch.randn_like(solution) * noise_level * max_amplitude
    noisy_solution = solution + noise

    return noisy_solution, solution

# 파라미터 설정
length = 1.0
mass = 1.0
gravity = 9.8
damping = 0.3
initial_state = torch.tensor([0.5, 0.0], dtype=torch.float32)  # 초기 각도 및 각속도
t_train = torch.linspace(0, 5, 1000)  # 0-5초 동안 1000 포인트
t_test = torch.linspace(5, 10, 1000)  # 5-10초 동안 1000 포인트
noise_level = 0.1  # 노이즈 수준

# 시스템 생성
system = DampedSinglePendulumSystem(length, mass, gravity, damping)

# 데이터 생성
train_data_noisy, train_data_clean = generate_noisy_data_rk4(system, initial_state, t_train, noise_level)
test_data, _ = generate_noisy_data_rk4(system, train_data_clean[-1], t_test, 0)  # 노이즈 없는 테스트 데이터

# 시퀀스 생성 함수
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# 시퀀스 데이터 준비
seq_length = 50
X_train = create_sequences(train_data_noisy, seq_length)
y_train = train_data_noisy[seq_length:]  # 다음 상태를 목표로 설정

# PyTorch 텐서로 변환
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

# 모델 초기화
input_size = 2
hidden_size = 50
output_size = 2
model = LSTMPendulum(input_size, hidden_size, output_size)

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# 모델 학습
n_epochs = 200
batch_size = 32

for epoch in range(n_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        outputs = model(batch_X)

        # 손실 계산
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

t = np.concatenate((t_train, t_test))  # 전체 시간 벡터 정의
model.eval()
with torch.no_grad():
    predictions = []
    # 초기 시퀀스를 초기값으로 패딩
    padded_input = np.tile(initial_state.numpy(), (seq_length, 1))
    input_seq = torch.FloatTensor(padded_input).unsqueeze(0)

    for i in range(len(t)):
        output = model(input_seq)
        predictions.append(output.squeeze().numpy())

        # 실제 데이터로 입력 시퀀스 업데이트
        if i < len(train_data_clean) - 1:
            new_data = torch.FloatTensor(train_data_clean[i + 1]).unsqueeze(0)
        else:
            new_data = output

        input_seq = torch.cat([input_seq[:, 1:, :], new_data.unsqueeze(1)], dim=1)

predictions = np.array(predictions)

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(t, np.concatenate((train_data_clean[:, 0], test_data[:, 0])), label='True', color='b')
plt.plot(t_train, train_data_noisy[:, 0], label='Train (Noisy)', alpha=0.7)
plt.plot(t, predictions[:, 0], '--', label='Predicted', color='r')
plt.axvline(x=5, color='g', linestyle='--', label='Train/Test Split')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Single Pendulum: True vs Predicted (Full Range with Padding)')
plt.show()