import deepxde as dde
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# PART 0: Physical Properties, Mode Flag, and Data Loading
# =============================================================================
# 모드 설정: chaotic 모드일 경우 True, moderate 모드일 경우 False
chaotic = False  
noise_level = "clean"   # 예: "clean", "noise_40%", 등

# 물성치 및 초기 조건 (chaotic 모드 기준)
if chaotic:
    m1 = 1.0
    m2 = 2.0
    L1 = 2.0
    L2 = 1.0
    d1_0 = np.pi        # 첫 번째 진자의 초기 각도
    v1_0 = 0.0          # 첫 번째 진자의 초기 각속도
    d2_0 = np.pi / 2    # 두 번째 진자의 초기 각도
    v2_0 = 0.0          # 두 번째 진자의 초기 각속도
    
else:
    m1 = 2.0
    m2 = 0.5
    L1 = 1.0
    L2 = 2.0
    d1_0 = 1.0          # 첫 번째 진자의 초기 각도
    v1_0 = 0.0          # 첫 번째 진자의 초기 각속도
    d2_0 = 0.5          # 두 번째 진자의 초기 각도
    v2_0 = 0.3          # 두 번째 진자의 초기 각속도
t_max = 5
g = 9.81
b1 = 0
b2 = 0

# 데이터 로딩 함수 (double pendulum)
def data_loading(noise_level, chaotic=False):
    """
    데이터 로드 함수 (double pendulum)
    Args:
        noise_level: 사용할 noise level (예: 'clean', 'noise_40%', ...)
        chaotic: True이면 chaotic 데이터, False이면 moderate 데이터 사용
    Returns:
        time_train, time_test, time_valid, train_data, test_data, valid_data
    """
    if chaotic:
        file_path = f"data/double_pendulum/chaotic/{noise_level}/"
    else:
        file_path = f"data/double_pendulum/moderate/{noise_level}/"
    
    time_train = np.loadtxt(f'{file_path}{noise_level}_input_train.txt', delimiter='\t', skiprows=1)
    time_test  = np.loadtxt(f'{file_path}{noise_level}_input_test.txt', delimiter='\t', skiprows=1)
    time_valid = np.loadtxt(f'{file_path}{noise_level}_input_valid.txt', delimiter='\t', skiprows=1)
    train_data = np.loadtxt(f'{file_path}{noise_level}_output_train.txt', delimiter='\t', skiprows=1)
    test_data  = np.loadtxt(f'{file_path}{noise_level}_output_test.txt', delimiter='\t', skiprows=1)
    valid_data = np.loadtxt(f'{file_path}{noise_level}_output_valid.txt', delimiter='\t', skiprows=1)
    
    return time_train, time_test, time_valid, train_data, test_data, valid_data

def preprocess_data(time_data, data):
    """
    시간 데이터와 나머지 데이터를 결합하여 시간순으로 정렬한 후,
    double pendulum 데이터의 출력은 다음과 같이 구성됨:
      [angle1, angular_velocity1, angular_acceleration1, angle2, angular_velocity2, angular_acceleration2]
    따라서, 오직 각가속도는 인덱스 3 (첫 번째 진자)와 6 (두 번째 진자)를 선택합니다.
    (단, 데이터 파일에는 첫 번째 열이 시간으로 포함됨)
    """
    combined = np.concatenate((time_data.reshape(-1, 1), data), axis=1)
    sorted_data = combined[np.argsort(combined[:, 0])]
    # sorted_data 컬럼: [time, angle1, angular_velocity1, angular_acceleration1, angle2, angular_velocity2, angular_acceleration2]
    return sorted_data[:, 0], sorted_data[:, [3, 6]]

# 데이터 로딩 수행
time_train_file, time_test_file, time_valid_file, train_data_file, test_data_file, valid_data_file = data_loading(noise_level, chaotic)
# 예: test 데이터의 시간과 각가속도 (두 진자에 대한)만 선택
time_test_sorted, test_angular_acc = preprocess_data(time_test_file, test_data_file)

# =============================================================================
# PART 1: PDE Definition & PINN Setup (DeepXDE) for Double Pendulum
# =============================================================================
def dy(t, x):
    return dde.grad.jacobian(x, t)

def pde(t, x):
    # mass 1 location
    x_1 = x[:, 0:1]
    # mass 2 location
    x_2 = x[:, 1:2]

    dx1_t = dde.grad.jacobian(x, t, i = 0, j = 0)
    dx2_t = dde.grad.jacobian(x, t, i = 1, j = 0)

    dx1_tt = dde.grad.hessian(x, t, i = 0, j = 0, component = 0)
    dx2_tt = dde.grad.hessian(x, t, i = 0, j = 0, component = 1)

    pde1 = (m1 + m2) * L1 * dx1_tt + m2 * L2 * dx2_tt * tf.cos(x_1-x_2) + m2 * L2 * dx2_t ** 2 * tf.sin(x_1-x_2) + (m1 + m2) * g * tf.sin(x_1)
    pde2 = m2 * L2 * dx2_tt + m2 * L1 * dx1_tt * tf.cos(x_1-x_2) - m2 * L1 * dx1_t ** 2 * tf.sin(x_1-x_2) + m2 * g * tf.sin(x_2)

    return [pde1, pde2]

def boundary_init(t, on_boundary):
    return on_boundary and np.isclose(t[0], 0)

geom = dde.geometry.Interval(0, t_max)

# 초기 조건: t=0에서 각 진자의 각도 설정
init_d1 = dde.icbc.PointSetBC(np.array([0]), np.array([d1_0]).reshape(-1, 1), component=0)
init_d2 = dde.icbc.PointSetBC(np.array([0]), np.array([d2_0]).reshape(-1, 1), component=1)
# 초기 속도 조건: 각 진자의 각속도
init_v1 = dde.OperatorBC(geom, lambda x, y, _: dy(x, y[:, 0:1]) - v1_0, boundary_init)
init_v2 = dde.OperatorBC(geom, lambda x, y, _: dy(x, y[:, 1:2]) - v2_0, boundary_init)

data = dde.data.PDE(geom,
                    pde,
                    [init_d1, init_d2, init_v1, init_v2],
                    num_domain = 2000,
                    num_boundary = 200,
                    num_test = 2000)

# 신경망(FNN) 구성: 입력은 시간 (1차원), 출력은 2차원 (각 진자의 각도)
layer_size = [1] + [64] * 10 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

# =============================================================================
# PART 2: Training with ModelCheckpoint Callback (Save Final Model Only)
# =============================================================================
epochs_adam = 20000
#save_model_dir = "example/trained_models/PINN_ckpt"
#os.makedirs(save_model_dir, exist_ok=True)
chaotic_str = "chaotic" if chaotic else "moderate"
# 체크포인트 콜백은 주기(period)마다 저장하지만, 여기서는 마지막에 저장되는 모델을 사용합니다.
#model_save_path = os.path.join(save_model_dir, f"Double_pen_{chaotic_str}_PINN_{noise_level}.ckpt")
# checker = dde.callbacks.ModelCheckpoint(
#     filepath=model_save_path,
#     verbose=1,
#     save_better_only=True,
#     period=20000  
# )

model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=epochs_adam) #callbacks=[checker])
dde.saveplot(losshistory, train_state, issave=False, isplot=False)

# LBFGS 단계 (선택 사항)
dde.optimizers.config.set_LBFGS_options(maxiter=15000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

print("Final training completed. The final model is available in 'model'.")

# Time span
t_span = np.linspace(0, 10, 2000)
t_test = np.linspace(0,10,2000)
t_reshaped = t_test.reshape(-1, 1)
result = model.predict(t_reshaped)
output_path = "Doublepen_predictions.npy"

t_grad = t_reshaped.reshape(-1)
# NumPy 배열로 저장
np.save(output_path, result)
usol1 = np.array(result[:, 0])
usol2 = np.array(result[:, 1])

print(t_grad.shape)


# Numerical differentiation to compute velocities (first derivative)
usol1_vel = np.gradient(usol1, t_grad)  # First derivative of theta1
usol2_vel = np.gradient(usol2, t_grad)  # First derivative of theta2

# Numerical differentiation to compute accelerations (second derivative)
usol1_accel = np.gradient(usol1_vel, t_grad)  # Second derivative of theta1 (acceleration)
usol2_accel = np.gradient(usol2_vel, t_grad)  # Second derivative of theta2 (acceleration)

# 중간 지점을 자동으로 계산
half = len(t_span) // 2

# Plotting the accelerations pred
plt.figure(figsize=(12, 6))

# Acceleration of theta1 (interpolation)
plt.plot(t_span[:half], usol1_accel[:half], label=r'$\ddot{\Theta}_1$ (PINN)', linestyle='dashed', color='r', lw=2)

# Acceleration of theta1 (extrapolation)
plt.plot(t_span[half:], usol1_accel[half:], label=r'$\ddot{\Theta}_1$ (PINN_ext)', linestyle='dashed', color='g', lw=2)

# Acceleration of theta2 (interpolation)
plt.plot(t_span[:half], usol2_accel[:half], label=r'$\ddot{\Theta}_2$ (PINN)', linestyle='dashed', color='b', lw=2)

# Acceleration of theta2 (extrapolation)
plt.plot(t_span[half:], usol2_accel[half:], label=r'$\ddot{\Theta}_2$ (PINN_ext)', linestyle='dashed', color='g', lw=2)


# Plotting the accelerations Reference

# Acceleration of theta1
plt.plot(t_span, test_angular_acc[:, 0], label=r'$\ddot{\Theta}_1$ (Ref)',  color='k', lw=2)

# Acceleration of theta2
plt.plot(t_span, test_angular_acc[:, 1], label=r'$\ddot{\Theta}_2$ (Ref)',  color='k', lw=2)

# Labels and plot settings
plt.legend(loc='best', fontsize=15)
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Acceleration (rad/s²)', fontsize=15)
plt.grid(True)

# plt.figure(figsize=(12, 6))
# plt.plot(time_test_sorted, test_angular_acc[:, 0], label="Test Acc (Pendulum 1, Ground Truth)", color="black", alpha=0.7)
# plt.plot(time_test_sorted, predicted_acc[:, 0], label="Predicted Acc (Pendulum 1)", color="red", linestyle="--")
# plt.plot(time_test_sorted, test_angular_acc[:, 1], label="Test Acc (Pendulum 2, Ground Truth)", color="black", alpha=0.7)
# plt.plot(time_test_sorted, predicted_acc[:, 1], label="Predicted Acc (Pendulum 2)", color="orange", linestyle="--")
# # 중앙 시간 기준
# extrapolation_start_time = time_test_sorted[len(time_test_sorted) // 2]
# plt.axvline(x=extrapolation_start_time, color='gray',linestyle='--', linewidth=2, label='Extrapolation Start')
# plt.xlabel("Time (s)", fontsize=20)
# plt.ylabel("Angular Acceleration (rad/s²)", fontsize=20)
# plt.legend(fontsize=15)
# plt.grid(True)

fig_dir = "example/figs"
os.makedirs(fig_dir, exist_ok=True)
fig_save_path = os.path.join(fig_dir, f"Double_pen_{chaotic_str}_PINN_{noise_level}.png")
plt.savefig(fig_save_path, dpi=200, bbox_inches="tight")
print(f"Figure saved to {fig_save_path}")

plt.show()

