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

# 플롯ting
plt.figure(figsize=(15, 15))

# 훈련 데이터 플롯
plt.subplot(3, 1, 1)
plt.scatter(train_sorted[:, 0], train_sorted[:, 1], label='Train Angle', color='b', s=10, alpha=0.5)  # 각도 산점도
plt.scatter(train_sorted[:, 0], train_sorted[:, 2], label='Train Angular Acceleration', color='r', s=10, alpha=0.5)  # 각가속도 산점도
plt.title('Training Data')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()

# 테스트 데이터 플롯
plt.subplot(3, 1, 2)
plt.scatter(test_sorted[:, 0], test_sorted[:, 1], label='Test Angle', color='b', s=10, alpha=0.5)  # 각도 산점도
plt.scatter(test_sorted[:, 0], test_sorted[:, 2], label='Test Angular Acceleration', color='r', s=10, alpha=0.5)  # 각가속도 산점도
plt.title('Test Data')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()

# 검증 데이터 플롯
plt.subplot(3, 1, 3)
plt.scatter(valid_sorted[:, 0], valid_sorted[:, 1], label='Valid Angle', color='b', s=10, alpha=0.5)  # 각도 산점도
plt.scatter(valid_sorted[:, 0], valid_sorted[:, 2], label='Valid Angular Acceleration', color='r', s=10, alpha=0.5)  # 각가속도 산점도
plt.title('Validation Data')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()

plt.tight_layout()  # 서브플롯 간격 조정
#plt.savefig(f'data/Dataset_ground_{noise_level}.png',dpi=200)  # 플롯 출력
plt.show()
