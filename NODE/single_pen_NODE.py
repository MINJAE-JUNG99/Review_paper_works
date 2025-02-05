import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm
import time
from torchdiffeq import odeint
import matplotlib.pyplot as plt

def rk4(func, t, dt, y):
    k1 = dt * func(t, y)
    k2 = dt * func(t + dt / 2, y + k1 / 2)
    k3 = dt * func(t + dt / 2, y + k2 / 2)
    k4 = dt * func(t + dt, y + k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6

class NeuralODE(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, y0, t, solver, Train=True):
        if Train:  # Training 모드일 때
            # y0: [32, 2], t: [20, 32]
            # solution will store the result with shape [20, 32, 2]
            solution = torch.empty(len(t), *y0.shape, dtype=y0.dtype, device=y0.device)

            # Initialize the first time step with y0 values for all 32 cases
            solution[0] = y0

            # Perform numerical integration for each y0 (32 in total)
            for i in range(y0.shape[0]):  # Loop over the 32 y0s
                y_current = y0[i]  # Current y0 for i-th case
                for j in range(1, len(t)):  # Loop over the time steps (20 time steps)
                    t0 = t[j-1, i]  # Starting time for i-th y0
                    t1 = t[j, i]    # Next time for i-th y0
                    delta_t = t1 - t0

                    # Perform numerical integration from t0 to t1 for i-th y0
                    dy = solver(self.func, t0, delta_t, y_current)
                    y_next = y_current + dy

                    # Store the result in the solution tensor
                    solution[j, i] = y_next

                    # Update y_current for the next step
                    y_current = y_next

        else:  # Evaluation 모드일 때
            # solution will store the result with shape [20, 32, 2]
            solution = torch.empty(len(t), *y0.shape, dtype=y0.dtype, device=y0.device)

            # Initialize the first time step with y0 values
            solution[0] = y0

            # Perform numerical integration for the entire batch at once
            for j in range(1, len(t)):  # Loop over the time steps (20 time steps)
                t0 = t[j-1]  # Starting time for this step
                t1 = t[j]    # Next time step
                delta_t = t1 - t0

                # Perform numerical integration from t0 to t1 for all y0s
                dy = solver(self.func, t0, delta_t, y0)
                y1 = y0 + dy

                # Store the result in the solution tensor
                solution[j] = y1

                # Update y0 for the next step
                y0 = y1

        return solution

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

# 시간과 데이터를 분리
time_train_sorted = train_sorted[:, 0]  # 첫 번째 열: 시간
data_train_sorted = train_sorted[:, 1:]  # 나머지 열: 데이터 (각도 및 각가속도)

time_test_sorted = test_sorted[:, 0]  # 첫 번째 열: 시간
data_test_sorted = test_sorted[:, 1:]  # 나머지 열: 데이터

time_valid_sorted = valid_sorted[:, 0]  # 첫 번째 열: 시간
data_valid_sorted = valid_sorted[:, 1:]  # 나머지 열: 데이터


# 시간과 데이터를 분리한 후 torch로 변환
time_train_sorted = torch.tensor(time_train_sorted, dtype=torch.float32)  # 시간 데이터 변환
data_train_sorted = torch.tensor(data_train_sorted, dtype=torch.float32)  # 각도 및 각가속도 데이터 변환
# print(data_train_sorted.size())
# print(time_train_sorted.size())

time_test_sorted = torch.tensor(time_test_sorted, dtype=torch.float32)    # 시간 데이터 변환
data_test_sorted = torch.tensor(data_test_sorted, dtype=torch.float32)    # 각도 및 각가속도 데이터 변환

time_valid_sorted = torch.tensor(time_valid_sorted, dtype=torch.float32)  # 시간 데이터 변환
data_valid_sorted = torch.tensor(data_valid_sorted, dtype=torch.float32)  # 각도 및 각가속도 데이터 변환


batch_time = 20
batch_size = 32
batch_data_size=800

train_batch=data_train_sorted[:,:]
def get_batch():
    s= torch.from_numpy(np.random.choice(np.arange(batch_data_size - batch_time, dtype=np.int64),batch_size, replace = False))
    batch_y0 = train_batch[s]
    batch_t = torch.stack([time_train_sorted[s+i] for i in range(batch_time)], dim = 0)
    batch_y = torch.stack([train_batch[s+i] for i in range(batch_time)], dim = 0)
    return batch_y0.cpu(), batch_t.cpu(), batch_y.cpu()

# def visualize():
#     # 학습 및 검증 손실 그래프
# plt.figure(figsize=(15, 10))


# plt.subplot(3, 1, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(valid_losses, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# # 테스트 데이터 예측 결과 - 각도
# plt.subplot(3, 1, 2)
# plt.plot(time_test_sorted, data_test_sorted[:, 0], label='True Angle', color='blue')
# plt.plot(time_test_sorted, predictions[:, 0], label='Predicted Angle', color='red', linestyle='--')
# plt.title('Test Data Prediction - Angle')
# plt.xlabel('Time')
# plt.ylabel('Angle')
# plt.legend()
# plt.grid(True)

# # 테스트 데이터 예측 결과 - 각가속도
# plt.subplot(3, 1, 3)
# plt.plot(time_test_sorted, data_test_sorted[:, 1], label='True Angular Acceleration', color='green')
# plt.plot(time_test_sorted, pred_y[:, 1], label='Predicted Angular Acceleration', color='orange', linestyle='--')
# plt.title('Test Data Prediction - Angular Acceleration')
# plt.xlabel('Time')
# plt.ylabel('Angular Acceleration')
# plt.legend()
# plt.grid(True)

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


node = NeuralODE(func=ODEFunc()).cpu()
optimizer = optim.Adam(node.parameters(), lr=1e-3)
niters = 2000
start_time = time.time()
loss_history = []
best_valid_loss = float('inf')  # Initialize with a large value
best_model = None  # To store the best model

for iter in tqdm(range(niters + 1)):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch()
    pred_y = node(y0=batch_y0, t=batch_t, solver=rk4, Train=True)
    loss = torch.mean(torch.square(pred_y - batch_y))
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if iter % 20 == 0:
        with torch.no_grad():
            # Predict on train and valid data
            pred_y_train = node(data_train_sorted[0, :], time_train_sorted, solver=rk4, Train=False)
            pred_y_valid = node(data_valid_sorted[0, :], time_valid_sorted, solver=rk4, Train=False)

            # Calculate losses
            loss_train = torch.mean(torch.abs(pred_y_train - data_train_sorted))
            loss_valid = torch.mean(torch.abs(pred_y_valid - data_valid_sorted))

            # Print losses
            print('Iter {:04d} | Train Loss {:6f},  Valid Loss {:6f}'.format(iter, loss_train.item(), loss_valid.item()))

            # Save model if valid loss improves
            if loss_valid.item() < best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_model = node.state_dict()  # Store the best model's state dict
                torch.save(best_model, 'NODE/runs/best_model.pt')  # Save the best model

            # Save intermediate prediction results
            if loss.item() < 0.2:  # 조건 추가
                filename = f'NODE/runs/pred_y_iter_{iter:04d}.pt'
                torch.save(pred_y_train, filename)

end_time = time.time() - start_time
print('Process time: {} sec'.format(end_time))

# Load the best model and evaluate on test data
best_model_path = 'NODE/runs/best_model.pt'
node.load_state_dict(torch.load(best_model_path))  # Load the best model

with torch.no_grad():
    # Predict on test data
    pred_y_test = node(data_test_sorted[0, :], time_test_sorted, solver=rk4, Train=False)

# Plot the test predictions
plt.figure(figsize=(10, 6))
plt.plot(time_test_sorted, data_test_sorted[:, 0], label='True Test Data')
plt.plot(time_test_sorted, pred_y_test[:, 0], label='Predicted Test Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Test Data vs. Predicted Data')
plt.legend()
plt.grid(True)
plt.show()

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss During Training')
plt.legend()
plt.grid(True)
plt.show()

# from torchdiffeq import odeint_adjoint

# niters = 3000

# func = ODEFunc().cpu()
# optimizer = optim.Adam(func.parameters(), lr=1e-3)

# start_time = time.time()
# loss_history=[]
# for iter in tqdm(range(niters+1)):
#     optimizer.zero_grad()
#     batch_y0, batch_t, batch_y = get_batch()
#     pred_y = odeint_adjoint(func=func, y0=batch_y0, t=batch_t, rtol=1e-7, atol=1e-9,method='dopri5')
#     loss = torch.mean(torch.abs(pred_y - batch_y))
#     loss.backward()
#     optimizer.step()

#     loss_history.append(loss.item())
#     if iter % 20 == 0:
#         with torch.no_grad():
#             pred_y = odeint_adjoint(func, true_y0, t, rtol=1e-7, atol=1e-9,method='dopri5')
#             loss = torch.mean(torch.abs(pred_y - true_y))
#             print('Iter {:04d} | Total Loss {:6f}'.format(iter, loss.item()))
#             visualize(true_y, true_y_noisy, pred_y)

# end_time = time.time() - start_time
# print('process time: {} sec'.format(end_time))

# # Plot loss history
# plt.figure(figsize=(10, 6))
# plt.plot(loss_history, label='Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Loss During Training')
# plt.legend()
# plt.grid(True)
# plt.show()