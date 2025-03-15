import os
import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =============================================================================
# PART 0: Physical Properties, Damping Flag, and Data Loading
# =============================================================================
# Set damping flag and noise level
damping = True          # True이면 감쇠, False이면 비감쇠
noise_level = "clean"   # 예: "clean", "noise_40%", 등

# 물성치 (damping 여부에 따라 c 값 변경)
t_max = 5
g = 9.81    # 중력 가속도 (m/s²)
m = 1       # 질량 (kg)
L = 1       # 진자 길이 (m)
if damping:
    c = 0.3  # 감쇠 계수 (damped)
else:
    c = 0.0  # 감쇠 없음 (undamped)

# 데이터 로딩 함수 (single pendulum)
def data_loading(noise_level, damping=False):
    """
    데이터 로드 함수
    Args:
        noise_level: 사용할 noise level (예: 'clean', 'noise_40%', ...)
        damping: True이면 감쇠 데이터, False이면 비감쇠 데이터 사용
    Returns:
        time_train, time_test, time_valid, train_data, test_data, valid_data
    """
    if damping:
        file_path = f"data/single_pendulum_damped/{noise_level}/"
    else:
        file_path = f"data/single_pendulum_undamped/{noise_level}/"
    
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
    출력 데이터는 오직 각가속도(angular acceleration)만 선택합니다.
    (원본 출력 데이터가 [angle, angular velocity, angular acceleration] 순서라고 가정)
    """
    combined = np.concatenate((time_data.reshape(-1, 1), data), axis=1)
    sorted_data = combined[np.argsort(combined[:, 0])]
    # 시간과 오직 마지막 열(각가속도)만 반환
    return sorted_data[:, 0], sorted_data[:, -1:]

# 데이터 로딩을 맨 위에서 수행
time_train_file, time_test_file, time_valid_file, train_data_file, test_data_file, valid_data_file = data_loading(noise_level, damping)
# 예를 들어, test 데이터의 시간과 각가속도만 선택
time_test_sorted, test_angular_acc = preprocess_data(time_test_file, test_data_file)

# =============================================================================
# PART 1: PDE Definition & PINN Setup (DeepXDE)
# =============================================================================
def dy(t, x):
    # x에 대한 t 방향의 1차 미분 (Jacobian)
    return dde.grad.jacobian(x, t)

def pde(t, x):
    """
    진자 운동에 대한 PDE:
      m*L*x_tt + m*g*sin(x) + (c*x_t)/(m*L) = 0
    (여기서 x는 진자의 각도)
    """
    x_1 = x[:, 0:1]  # 예시: x의 첫 성분 (각도)
    dx1_t = dde.grad.jacobian(x, t, i=0, j=0)
    dx1_tt = dde.grad.hessian(x, t, i=0, j=0, component=0)
    residual = m * L * dx1_tt + g * m * tf.math.sin(x_1) + (c * dx1_t) / (m * L)
    return residual

def boundary_init(t, on_boundary):
    # 초기 시각 t=0에서 경계 조건 적용
    return on_boundary and np.isclose(t[0], 0)

# 초기 조건: t=0에서 각도 = x_0
x_0 = 0.5  # 초기 각도
init_theta = dde.icbc.PointSetBC(
    np.array([0]),
    np.array([x_0]).reshape(-1, 1),
    component=0
)

# 초기 각속도 조건: 예시로 OperatorBC (실제 v_0=0에 맞게 수정 가능)
v_0 = 0.5
init_theta_dot = dde.OperatorBC(
    dde.geometry.Interval(0, t_max),
    lambda x, y, _: dy(x, y[:, 0:1]) - v_0,
    boundary_init
)

# 정의역: 시간 구간 [0, t_max]
geom = dde.geometry.Interval(0, t_max)

# DeepXDE 데이터 객체 생성 (PDE, 경계조건, 도메인 포인트 등)
data = dde.data.PDE(
    geom,
    pde,
    [init_theta, init_theta_dot],
    num_domain=2000,
    num_boundary=100,
    num_test=1000
)

# 신경망(FNN) 모델 구성
layer_size = [1] + [64] * 5 + [1]
activation = "Tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

# DeepXDE 모델 생성
model = dde.Model(data, net)

epochs_adam = 20000
save_model_dir = "example/trained_models/PINN_ckpt"
os.makedirs(save_model_dir, exist_ok=True)
damped_str = "damped" if damping else "undamped"
model_save_path = os.path.join(save_model_dir, f"single_pen_{damped_str}_PINN_{noise_level}.ckpt")
checker = dde.callbacks.ModelCheckpoint(
    filepath=model_save_path, 
    verbose=1, 
    save_better_only=True,  
    period=20000
)


model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=epochs_adam, callbacks=[checker])
dde.saveplot(losshistory, train_state, issave=False, isplot=False)

# LBFGS 단계에서 추가 학습 (여기서는 checkpoint callback 없이 진행)
dde.optimizers.config.set_LBFGS_options(maxiter=15000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=False)

# =============================================================================
# PART 2: Computing Predicted Angular Acceleration via PINN using np.gradient
# =============================================================================
def model_angular_acceleration_np(t_np):
    """
    모델 예측값 (각도)을 np.gradient를 사용해 2계 도함수를 계산하여 
    각가속도를 구하는 함수입니다.
    
    Args:
        t_np: 시간 샘플 (numpy array, shape: (N,1) 또는 (N,))
        
    Returns:
        u_tt: 각가속도 (numpy array, shape: (N,))
    """
    # model.predict는 np.array 반환 (예: shape (N,1))
    u = model.predict(t_np)
    u = u.flatten()          # 1D 배열로 변환
    t = t_np.flatten()       # 시간도 1D 배열로 변환

    # np.gradient를 이용해 1차 도함수 계산
    u_t = np.gradient(u, t)  
    # 다시 np.gradient로 2차 도함수 계산
    u_tt = np.gradient(u_t, t)
    
    return u_tt


# PINN을 통해 test 데이터 시간에서 예측한 각가속도 계산
predicted_acc = model_angular_acceleration_np(time_test_sorted.reshape(-1, 1))

# 결과 플롯 (ground truth vs. model prediction)
plt.figure(figsize=(12, 6))
plt.plot(time_test_sorted, test_angular_acc, label="Test Angular Acceleration (Ground Truth)", color="black")
plt.plot(time_test_sorted, predicted_acc, label="Predicted Angular Acceleration", color="red", linestyle="--")
# 중앙 시간 기준
extrapolation_start_time = time_test_sorted[len(time_test_sorted) // 2]
plt.axvline(x=extrapolation_start_time, color='gray',linestyle='--', linewidth=2, label='Extrapolation Start')
plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("Angular Acceleration (rad/s²)", fontsize=20)
plt.legend(fontsize=10)
plt.grid(True)

# 플롯 저장
fig_dir = "example/figs"
os.makedirs(fig_dir, exist_ok=True)
fig_save_path = os.path.join(fig_dir, f"single_pen_{damped_str}_PINN_{noise_level}.png")
plt.savefig(fig_save_path, dpi=200, bbox_inches='tight')
print(f"Figure saved to {fig_save_path}")

plt.show()

# IF you want to load trained model ckpt to pred
# model.restore(model_save_path) # model ckpt 복원 과정에서 저장경로 뒤에 숫자가 붙어나오므로 저장된 모델 파일 확인후 복원하는 과정이 필요함
# predicted_acc = model_angular_acceleration_np(time_test_sorted.reshape(-1, 1))