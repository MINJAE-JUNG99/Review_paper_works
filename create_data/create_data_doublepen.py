import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

# ODE function
def double_pendulum_ode(y, t, m1, m2, L1, L2, g):
    dydt = np.zeros(4)

    # Extracting variables
    theta_1, omega_1, theta_2, omega_2 = y  # Changed variable names

    # Equations of motion
    dydt[0] = omega_1  # d(theta_1)/dt = omega_1
    dydt[1] = (-g * (2 * m1 + m2) * np.sin(theta_1) - m2 * g * np.sin(theta_1 - 2 * theta_2) - 
                2 * np.sin(theta_1 - theta_2) * m2 * (omega_2**2 * L2 + omega_1**2 * L1 * np.cos(theta_1 - theta_2))) / \
               (L1 * (2 * m1 + m2 - m2 * np.cos(2 * theta_1 - 2 * theta_2)))
    dydt[2] = omega_2  # d(theta_2)/dt = omega_2
    dydt[3] = (2 * np.sin(theta_1 - theta_2) * (omega_1**2 * L1 * (m1 + m2) + 
                g * (m1 + m2) * np.cos(theta_1) + omega_2**2 * L2 * m2 * np.cos(theta_1 - theta_2))) / \
               (L2 * (2 * m1 + m2 - m2 * np.cos(2 * theta_1 - 2 * theta_2)))

    return dydt


is_chaotic = False
uniform = True

# Parameters for double pendulum
if is_chaotic:
    m1 = 1.0   # mass of first pendulum in chaotic state
    m2 = 2.0   # mass of second pendulum in chaotic state
    L1 = 2.0   # length of first pendulum in chaotic state
    L2 = 1.0   # length of second pendulum in chaotic state
    g = 9.81

    # Initial conditions for chaotic state
    theta1_0 = np.pi  # initial angle1 for chaotic state (in radians)
    omega1_0 = 0.0  # initial angular velocity1 for chaotic state
    theta2_0 = np.pi/2  # initial angle2 for chaotic state (in radians)
    omega2_0 = 0.0  # initial angular velocity2 for chaotic state
else:
    m1 = 2.0   # mass of first pendulum in moderate state
    m2 = 0.5   # mass of second pendulum in moderate state
    L1 = 1.0   # length of first pendulum in moderate state
    L2 = 2.0   # length of second pendulum in moderate state
    g = 9.81

    # Initial conditions for moderate state
    theta1_0 = 1.0  # initial angle1 for moderate state (in radians)
    omega1_0 = 0.0  # initial angular velocity1 for moderate state
    theta2_0 = 0.5  # initial angle2 for moderate state (in radians)
    omega2_0 = 0.3  # initial angular velocity2 for moderate state

# 초기 조건을 리스트로 설정
initial_conditions = [theta1_0, omega1_0, theta2_0, omega2_0]

# ODE solver 설정
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10  # 전체 시간은 10초
numpoints = 8010  # 5000개의 데이터를 생성
t = np.linspace(0, stoptime, numpoints)

# ODE 풀기
sol = odeint(double_pendulum_ode, initial_conditions, t, args=(m1, m2, L1, L2, g))

# Extracting angles and angular velocities
theta1 = sol[:, 0]
omega1 = sol[:, 1]
theta2 = sol[:, 2]
omega2 = sol[:, 3]

# Angular accelerations using numerical differentiation
alpha1 = np.gradient(omega1, t)  # Angular acceleration for first pendulum
alpha2 = np.gradient(omega2, t)  # Angular acceleration for second pendulum

# 노이즈 추가
noise_level = 0.0
if noise_level == 0.0:
    dict_data = 'clean'
else:
    dict_data = f'noise_{int(noise_level * 100)}%'

theta_1_amp = np.max(theta1)
omega_1_amp = np.max(omega1)
alpha_1_amp = np.max(alpha1)
theta_2_amp = np.max(theta2)
omega_2_amp = np.max(omega2)
alpha_2_amp = np.max(alpha2)

# 노이즈 추가
theta1_noise = np.random.normal(0, noise_level * theta_1_amp, size=theta1.shape)
alpha1_noise = np.random.normal(0, noise_level * alpha_1_amp, size=alpha1.shape)
omega1_noise = np.random.normal(0, noise_level * omega_1_amp, size=omega1.shape)
theta2_noise = np.random.normal(0, noise_level * theta_2_amp, size=theta2.shape)
alpha2_noise = np.random.normal(0, noise_level * alpha_2_amp, size=alpha2.shape)
omega2_noise = np.random.normal(0, noise_level * omega_2_amp, size=omega2.shape)

# 노이즈가 추가된 값
theta1_noisy = theta1 + theta1_noise
omega1_noisy = omega1 + omega1_noise
alpha1_noisy = alpha1 + alpha1_noise
theta2_noisy = theta2 + theta2_noise
omega2_noisy = omega2 + omega2_noise
alpha2_noisy = alpha2 + alpha2_noise

# 시간, 각도, 각가속도를 튜플로 묶음
data = np.array(list(zip(t, theta1_noisy, omega1_noisy, alpha1_noisy, theta2_noisy, omega2_noisy, alpha2_noisy)))
if uniform:
    train_size = 800
    timestep = 5
    valid_size = 200
    test_size = 1600
    # 학습 데이터 인덱스 (0부터 시작)
    train_indices = np.arange(0, train_size * timestep, timestep)
        
    # 검증 데이터 인덱스 (offset = 1로 두어서 안 겹치도록)
    valid_start = 1
    val_indices = np.arange(valid_start, valid_start + valid_size * timestep * 4, timestep*4)
        
    # 테스트 데이터 인덱스 (offset = 2로 두어서 안 겹치도록)
    test_start = 2
    test_indices = np.arange(test_start, test_start + test_size * timestep, timestep)
        
        
    train_data = data[train_indices]
    val_data = data[val_indices]
    test_data = data[test_indices]
else:
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
plt.figure(figsize=(12, 12))

# Train 데이터 플롯 (θ, ω, α)
plt.subplot(6, 1, 1)
plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', s=1, label='θ1 (Train)')
plt.scatter(train_data[:, 0], train_data[:, 4], color='orange', s=1, label='θ2 (Train)')
plt.title("Train Data (Angles)")
plt.xlabel("Time (s)")
plt.ylabel("Angle (radians)")
plt.legend()
plt.grid()

plt.subplot(6, 1, 2)
plt.scatter(train_data[:, 0], train_data[:, 2], color='cyan', s=1, label='ω1 (Train)')
plt.scatter(train_data[:, 0], train_data[:, 5], color='red', s=1, label='ω2 (Train)')
plt.title("Train Data (Angular Velocities)")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.grid()

plt.subplot(6, 1, 3)
plt.scatter(train_data[:, 0], train_data[:, 3], color='purple', s=1, label='α1 (Train)')
plt.scatter(train_data[:, 0], train_data[:, 6], color='green', s=1, label='α2 (Train)')
plt.title("Train Data (Angular Accelerations)")
plt.xlabel("Time (s)")
plt.ylabel("Angular Acceleration (rad/s²)")
plt.legend()
plt.grid()

# Validation 데이터 플롯 (θ, ω, α)
plt.subplot(6, 1, 4)
plt.scatter(val_data[:, 0], val_data[:, 1], color='blue', s=1, label='θ1 (Validation)')
plt.scatter(val_data[:, 0], val_data[:, 4], color='orange', s=1, label='θ2 (Validation)')
plt.title("Validation Data (Angles)")
plt.xlabel("Time (s)")
plt.ylabel("Angle (radians)")
plt.legend()
plt.grid()

plt.subplot(6, 1, 5)
plt.scatter(val_data[:, 0], val_data[:, 2], color='cyan', s=1, label='ω1 (Validation)')
plt.scatter(val_data[:, 0], val_data[:, 5], color='red', s=1, label='ω2 (Validation)')
plt.title("Validation Data (Angular Velocities)")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.grid()

plt.subplot(6, 1, 6)
plt.scatter(val_data[:, 0], val_data[:, 3], color='purple', s=1, label='α1 (Validation)')
plt.scatter(val_data[:, 0], val_data[:, 6], color='green', s=1, label='α2 (Validation)')
plt.title("Validation Data (Angular Accelerations)")
plt.xlabel("Time (s)")
plt.ylabel("Angular Acceleration (rad/s²)")
plt.legend()
plt.grid()

# 현재 작업 디렉토리 확인 및 저장 경로 설정
if is_chaotic:
    if uniform:
        save_dir = os.path.join(os.getcwd(), 'Double_pen_chaotic_uniform', dict_data)
    else:
        save_dir = os.path.join(os.getcwd(), 'Double_pen_chaotic', dict_data)
    
else:
    if uniform:
        save_dir = os.path.join(os.getcwd(), 'Double_pen_moderate_uniform', dict_data)
    else:
        save_dir = os.path.join(os.getcwd(), 'Double_pen_moderate', dict_data)

os.makedirs(save_dir, exist_ok=True) 
plt.savefig(os.path.join(save_dir, 'double_pendulum_data_plot.png'), dpi=300, bbox_inches='tight')

# Saving the input data (time only for input files)
np.savetxt(os.path.join(save_dir, f'{dict_data}_chaotic_input_train.txt' if is_chaotic else f'{dict_data}_moderate_input_train.txt'), 
           np.column_stack((train_data[:, 0],)), header='time', delimiter='\t')
np.savetxt(os.path.join(save_dir, f'{dict_data}_chaotic_input_valid.txt' if is_chaotic else f'{dict_data}_moderate_input_valid.txt'), 
           np.column_stack((val_data[:, 0],)), header='time', delimiter='\t')
np.savetxt(os.path.join(save_dir, f'{dict_data}_chaotic_input_test.txt' if is_chaotic else f'{dict_data}_moderate_input_test.txt'), 
           np.column_stack((test_data[:, 0],)), header='time', delimiter='\t')

# Saving the output data (including theta, omega, and alpha for both pendulums)
np.savetxt(os.path.join(save_dir, f'{dict_data}_chaotic_output_train.txt' if is_chaotic else f'{dict_data}_moderate_output_train.txt'), 
           np.column_stack((train_data[:, 1], train_data[:, 3], train_data[:, 2], train_data[:, 4], train_data[:, 6], train_data[:, 5])), 
           header='theta1\tomega1\talpha1\ttheta2\tomega2\talpha2', delimiter='\t')

np.savetxt(os.path.join(save_dir, f'{dict_data}_chaotic_output_valid.txt' if is_chaotic else f'{dict_data}_moderate_output_valid.txt'), 
           np.column_stack((val_data[:, 1], val_data[:, 3], val_data[:, 2], val_data[:, 4], val_data[:, 6], val_data[:, 5])), 
           header='theta1\tomega1\talpha1\ttheta2\tomega2\talpha2', delimiter='\t')

np.savetxt(os.path.join(save_dir, f'{dict_data}_chaotic_output_test.txt' if is_chaotic else f'{dict_data}_moderate_output_test.txt'), 
           np.column_stack((test_data[:, 1], test_data[:, 3], test_data[:, 2], test_data[:, 4], test_data[:, 6], test_data[:, 5])), 
           header='theta1\tomega1\talpha1\ttheta2\tomega2\talpha2', delimiter='\t')

# import pandas as pd
# output_df = pd.read_csv('Double_pen_chaotic_uniform/clean/clean_chaotic_output_test.txt', delimiter='\t')
# print(output_df.head())  # 상위 몇 개 행을 출력하여 열 구분 확인

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
