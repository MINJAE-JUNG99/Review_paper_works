import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

# ODE function for single pendulum with damping
def single_pendulum_ode(y, t, m, L, g, c):
    dydt = np.zeros(2)
    
    # Extracting variables
    theta, omega = y  # theta: angle, omega: angular velocity
    
    # Equations of motion with damping (theta'' + (b/m) * omega + (g/L) * sin(theta) = 0)
    dydt[0] = omega  # d(theta)/dt = omega
    dydt[1] = -(g / L) * np.sin(theta) - (c / m) * omega  # d(omega)/dt with damping
    
    return dydt

# Parameters for single pendulum
m = 1.0   # mass
L = 1.0   # length
g = 9.81  # gravity
c = 0.0   # damping coefficient
noise_level = 0.0
if noise_level == 0.0:
    dict_data = 'clean'
else:
    dict_data = f'noise_{int(noise_level * 100)}%'  # Noise level에 따라 동적으로 설정

# Initial conditions
theta_0 = 0.5  # initial angle (45 degrees)
omega_0 = 0.5        # initial angular velocity
initial_conditions = [theta_0, omega_0]

# ODE solver 설정
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10  # 전체 시간은 10초
numpoints = 5000  # 5000개의 데이터를 생성
t = np.linspace(0, stoptime, numpoints)

# ODE 풀기
sol = odeint(single_pendulum_ode, initial_conditions, t, args=(m, L, g, c))

# Extracting angles and angular velocities
theta = sol[:, 0]
omega = sol[:, 1]
alpha = np.gradient(omega, t)  # Angular acceleration

# 노이즈 추가 (10%의 진폭)
theta_amplitude = np.max(theta)  # 각도 진폭의 최대값
omega_amplitude = np.max(omega)
alpha_amplitude = np.max(np.abs(alpha))  # 각속도 진폭의 최대값

# 노이즈 생성
theta_noise = np.random.normal(0, noise_level * theta_amplitude, size=theta.shape)
omega_noise = np.random.normal(0, noise_level * omega_amplitude, size=omega.shape)
alpha_noise = np.random.normal(0, noise_level * alpha_amplitude, size=alpha.shape)

# 노이즈가 추가된 각도 및 각속도
theta_noisy = theta + theta_noise
omega_noisy = omega + omega_noise
alpha_noisy = alpha + alpha_noise


# Angular accelerations using numerical differentiation


# 시간, 각도, 각가속도를 튜플로 묶음
data = np.array(list(zip(t, theta_noisy, omega_noisy, alpha_noisy)))

# Train과 Validation 데이터는 0 ~ 5초 구간에서 선택
train_val_data = data[t <= 5]
train_val_indices = np.random.choice(len(train_val_data), 1000, replace=False)  # 1000개의 데이터 랜덤 선택

# Train: 80%, Validation: 20%로 나누기
train_size = int(0.8 * 1000)  # 800개의 Train 데이터
train_indices = np.random.choice(train_val_indices, train_size, replace=False)
val_indices = np.setdiff1d(train_val_indices, train_indices)

train_data = train_val_data[train_indices]
val_data = train_val_data[val_indices]

# Test 데이터는 0 ~ 10초 구간에서 선택하되, 앞선 Train/Validation 데이터와 겹치지 않게 설정
test_data_range_ext = data[(t > 5) & (t <= 10)]  # 5초에서 10초 구간의 데이터 선택

# 0초에서 5초 구간의 데이터 선택
test_data_range_int = data[(t <= 5)]  # 0초에서 5초 구간의 데이터 선택

# 5초에서 10초 구간에서 500개 랜덤으로 선택
test_indices_ext = np.random.choice(len(test_data_range_ext), 500, replace=False) 
test_data_ext = test_data_range_ext[test_indices_ext]  # 5초에서 10초 구간의 데이터 추출

# 0초에서 5초 구간의 데이터 중 Train/Validation과 겹치지 않는 인덱스 선택
remaining_indices_int = np.setdiff1d(np.arange(len(test_data_range_int)), train_val_indices)  # Train/Validation과 겹치지 않는 인덱스
test_indices_int = np.random.choice(remaining_indices_int, 500, replace=False)  # 0에서 5초 구간 내에서 겹치지 않는 500개 선택
test_data_int = test_data_range_int[test_indices_int]  # 0초에서 5초 구간의 데이터 추출

# 최종 테스트 데이터는 두 구간의 데이터를 결합
test_data = np.vstack((test_data_ext, test_data_int))

# 플롯 그리기
plt.figure(figsize=(12, 10))

# Train 데이터 플롯
plt.subplot(3, 1, 1)
plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', s=1, label='θ (Train)')
plt.scatter(train_data[:, 0], train_data[:, 2], color='purple', s=1, label='ω (Train)')
plt.scatter(train_data[:, 0], train_data[:, 3], color='cyan', s=1, label='α (Train)')
plt.title("Train Data (Angle, Angular Velocity, and Angular Acceleration)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid()

# Validation 데이터 플롯
plt.subplot(3, 1, 2)
plt.scatter(val_data[:, 0], val_data[:, 1], color='orange', s=1, label='θ (Validation)')
plt.scatter(val_data[:, 0], val_data[:, 2], color='purple', s=1, label='ω (Validation)')
plt.scatter(val_data[:, 0], val_data[:, 3], color='red', s=1, label='α (Validation)')
plt.title("Validation Data (Angle, Angular Velocity, and Angular Acceleration)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid()

# Test 데이터 플롯
plt.subplot(3, 1, 3)
plt.scatter(test_data[:, 0], test_data[:, 1], color='green', s=1, label='θ (Test)')
plt.scatter(test_data[:, 0], test_data[:, 2], color='purple', s=1, label='ω (Test)')
plt.scatter(test_data[:, 0], test_data[:, 3], color='lime', s=1, label='α (Test)')
plt.title("Test Data (Angle, Angular Velocity, and Angular Acceleration)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid()

# 현재 작업 디렉토리 확인 및 저장 경로 설정
save_dir = os.path.join(os.getcwd(), 'single_pendulum_chaotic', dict_data)
os.makedirs(save_dir, exist_ok=True) 
plt.savefig(os.path.join(save_dir, 'pendulum_data_plot.png'), dpi=300, bbox_inches='tight')

# 현재 작업 디렉토리 확인 및 저장 경로 설정
save_dir = os.path.join(os.getcwd(), 'single_pendulum_chaotic', dict_data)
os.makedirs(save_dir, exist_ok=True) 
plt.savefig(os.path.join(save_dir, 'pendulum_data_plot.png'), dpi=300, bbox_inches='tight')

# Saving the input data
np.savetxt(os.path.join(save_dir, f'{dict_data}_input_train.txt'), np.column_stack((train_data[:, 0],)), header='time', delimiter='\t')
np.savetxt(os.path.join(save_dir, f'{dict_data}_input_valid.txt'), np.column_stack((val_data[:, 0],)), header='time', delimiter='\t')
np.savetxt(os.path.join(save_dir, f'{dict_data}_input_test.txt'), np.column_stack((test_data[:, 0],)), header='time', delimiter='\t')

# Saving the output data (각도, 각속도, 각가속도 포함)
np.savetxt(os.path.join(save_dir, f'{dict_data}_output_train.txt'), np.column_stack((train_data[:, 1], train_data[:, 2], train_data[:, 3])), header='angle\tangular_velocity\tangular_acceleration', delimiter='\t')
np.savetxt(os.path.join(save_dir, f'{dict_data}_output_valid.txt'), np.column_stack((val_data[:, 1], val_data[:, 2], val_data[:, 3])), header='angle\tangular_velocity\tangular_acceleration', delimiter='\t')
np.savetxt(os.path.join(save_dir, f'{dict_data}_output_test.txt'), np.column_stack((test_data[:, 1], test_data[:, 2], test_data[:, 3])), header='angle\tangular_velocity\tangular_acceleration', delimiter='\t')

# 각 데이터 세트의 시간 값 추출
train_times = train_data[:, 0]  # Train 데이터의 시간
val_times = val_data[:, 0]      # Validation 데이터의 시간
test_times = test_data[:, 0]    # Test 데이터의 시간

# 겹치는 데이터 확인
train_val_overlap = np.intersect1d(train_times, val_times)  # Train과 Validation 간의 겹침
train_test_overlap = np.intersect1d(train_times, test_times)  # Train과 Test 간의 겹침
val_test_overlap = np.intersect1d(val_times, test_times)  # Validation과 Test 간의 겹침

# 결과 출력
if len(train_val_overlap) == 0:
    print("Train과 Validation 데이터 간에는 겹치는 데이터가 없습니다.")
else:
    print("Train과 Validation 데이터 간에 겹치는 데이터가 있습니다:", train_val_overlap)

if len(train_test_overlap) == 0:
    print("Train과 Test 데이터 간에는 겹치는 데이터가 없습니다.")
else:
    print("Train과 Test 데이터 간에 겹치는 데이터가 있습니다:", train_test_overlap)

if len(val_test_overlap) == 0:
    print("Validation과 Test 데이터 간에는 겹치는 데이터가 없습니다.")
else:
    print("Validation과 Test 데이터 간에 겹치는 데이터가 있습니다:", val_test_overlap)

