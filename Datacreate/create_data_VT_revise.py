import numpy as np
import matplotlib.pyplot as plt
import os

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def add_noise(data, noise_level=0.0, seed=42):
    """
    데이터에 가우시안 노이즈를 추가합니다.
    
    Args:
        data: 원본 데이터 (NumPy 배열, shape: (n, features))
        noise_level: 노이즈 수준 (0.0 ~ 1.0), 최대 진폭의 비율로 표준편차 결정
        seed: 재현성을 위한 랜덤 시드
    Returns:
        noisy_data: 노이즈가 추가된 데이터
    """
    if noise_level <= 0:
        return data.copy()
    
    np.random.seed(seed)
    noisy_data = np.empty_like(data)
    for i in range(data.shape[1]):
        max_amp = np.max(np.abs(data[:, i]))
        noise = np.random.normal(0, noise_level * max_amp, size=data[:, i].shape)
        noisy_data[:, i] = data[:, i] + noise
    return noisy_data

#정규화 함수 추가
def normalize_data(data, method="z-score", return_scaler=False):
    
    if method == "z-score":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / (std + 1e-8)
        scaler = {'method': 'z-score', 'mean': mean, 'std': std}
    elif method == "min-max":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
        scaler = {'method': 'min-max', 'min': min_val, 'max': max_val}
    else:
        raise ValueError("method는 'z-score' 또는 'min-max' 중 하나여야 합니다.")
    
    if return_scaler:
        return normalized_data, scaler
    else:
        return normalized_data

#########################################
# 1. CSV 파일 로드 및 데이터 전처리
#########################################
# noise_level 설정: 0.0이면 노이즈 없음, 0.1이면 최대 진폭의 10% 노이즈 추가
for noise_level in [0.0, 0.1]:
    # 각 noise_level마다 각 dynamics의 스케일러를 저장할 딕셔너리 생성
    scalers = {}  # key: dynamics_name, value: scaler dict

    # Time와 Dynamics 딕셔너리: 각 데이터셋(학습, 검증, 테스트)에 대해 각 dynamics 데이터를 저장
    Time = {k: {} for k in ['train', 'valid', 'test']}
    Dynamics = {k: {} for k in ['train', 'valid', 'test']}

    for dynamics_name in ['Pos', 'Vel', 'Acc']:
        # CSV 파일 로드 (예: 25,000행 데이터가 있다고 가정)
        file_path = "VT/VT_Dense_PointA_%s.csv" % dynamics_name
        data = np.genfromtxt(file_path, delimiter=",", skip_header=1)

        # 데이터 컬럼: [No, TIME, Acc_TX@StRq3, Acc_TY@StRq3, Acc_TZ@StRq3]
        time_all = data[:, 1]         # TIME 열, shape: (n_total,)
        dynamics = data[:, 2:5]         # dynamics 데이터, shape: (n_total, 3)
        n_total = data.shape[0]  # 예: 25000
        
        # 정규화: dynamics 데이터를 z-score 정규화 (clean 데이터 기준)
        # 정규화와 함께 스케일러를 반환받습니다.
        dynamics, scaler = normalize_data(dynamics, method="z-score", return_scaler=True)
        # 각 dynamics별 스케일러 저장 (나중에 복원에 사용)
        scalers[dynamics_name] = scaler

        # 파라미터 설정
        timestep = 5        # 인덱스 간격
        train_size = 2500   # 학습 데이터 샘플 개수
        valid_size = 250    # 검증 데이터 샘플 개수
        test_size  = 4999   # 테스트 데이터 샘플 개수

        # 학습 데이터 인덱스: 0부터 시작, 일정한 timestep 간격
        train_indices = np.arange(0, train_size * timestep, timestep)
        # 검증 데이터 인덱스: 학습 데이터와 겹치지 않도록 offset=1 사용, 간격은 timestep*10 (예시)
        valid_start = 1
        valid_indices = np.arange(valid_start, valid_start + valid_size * timestep * 5, timestep * 5)
        # 테스트 데이터 인덱스: 전체 데이터에서 학습 및 검증에 사용되지 않은 인덱스 중 offset=2 사용
        test_start = 2
        test_indices = np.arange(test_start, test_start + test_size * timestep, timestep)

        # 각 데이터셋 분할 (dynamics_name에 해당하는 데이터를 저장)
        J = dynamics_name
        Time['train'][J] = time_all[train_indices]
        Dynamics['train'][J] = dynamics[train_indices]

        Time['valid'][J] = time_all[valid_indices]
        Dynamics['valid'][J] = dynamics[valid_indices]

        Time['test'][J] = time_all[test_indices]
        Dynamics['test'][J] = dynamics[test_indices]

        # 데이터셋 간 겹침 여부 확인
        intersect_train_val = np.intersect1d(train_indices, valid_indices)
        intersect_train_test = np.intersect1d(train_indices, test_indices)
        intersect_val_test = np.intersect1d(valid_indices, test_indices)

        print(f"[{dynamics_name}] Intersection between training and validation indices:", intersect_train_val)
        print(f"[{dynamics_name}] Intersection between training and test indices:", intersect_train_test)
        print(f"[{dynamics_name}] Intersection between validation and test indices:", intersect_val_test)

        if (intersect_train_val.size == 0 and 
            intersect_train_test.size == 0 and 
            intersect_val_test.size == 0):
            print(f"[{dynamics_name}] No overlapping indices among training, validation, and test datasets.")
        else:
            print(f"[{dynamics_name}] There is overlap in the datasets.")

        # 노이즈 추가 설정 (예: 10% 노이즈)
        if noise_level > 0.:
            # 학습과 검증 데이터에만 노이즈 추가 (테스트 데이터는 그대로 사용)
            for k in ['train', 'valid']:
                Dynamics[k][J] = add_noise(Dynamics[k][J], noise_level=noise_level, seed=42)
    
    # 스케일러 저장: 각 dynamics에 대한 scaler를 하나의 dict로 저장
    scaler_save_dir = os.path.join("example", "scaler")
    os.makedirs(scaler_save_dir, exist_ok=True)
    noise_str = {0.0: "clean", 0.1: "noise10%"}[noise_level]
    scaler_save_path = os.path.join(scaler_save_dir, f"scaler_MBD_VT_{noise_str}.pkl")
    with open(scaler_save_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to: {scaler_save_path}")
    
    # 데이터 저장: 폴더 이름은 노이즈 수준에 따라 결정 (예: clean 또는 noise10%)
    folder = './VT/%s/' % ({0.0: 'clean', 0.1: 'noise10%'}[noise_level])
    os.makedirs(folder, exist_ok=True)
    
    for k in ['train', 'valid', 'test']:
        # 시간 데이터 저장 (2차원 배열, shape: (n,1))
        np.savetxt(folder + 'input_%s.txt' % k, Time[k]['Acc'].reshape(-1, 1), delimiter='\t', header="Time", comments="")
        # 가속도 데이터 저장: 3축 데이터를 순서대로 쌓기 
        dynamics_all = np.hstack([Dynamics[k][j] for j in ['Pos', 'Vel', 'Acc']])
        # 재정렬: [Pos, Vel, Acc] -> [u, dot{u}, ddot{u}, v, dot{v}, ddot{v}, w, dot{w}, ddot{w}]
        dynamics_all = np.hstack([dynamics_all[:, j].reshape(-1, 1) for j in [0, 3, 6, 1, 4, 7, 2, 5, 8]])
        assert(dynamics_all.shape[1] == 9)
        np.savetxt(folder + 'output_%s.txt' % k, dynamics_all, delimiter='\t',
                   header="Pos_TX\tVel_TX\tAcc_TX\tPos_TY\tVel_TY\tAcc_TY\tPos_TZ\tVel_TZ\tAcc_TZ", comments="")


# # 모든 데이터셋 플롯 (세 축 모두 플롯)
# acc_labels = ["Acc_TX@StRq3", "Acc_TY@StRq3", "Acc_TZ@StRq3"]

# fig, axs = plt.subplots(3, 1, figsize=(12, 18))
# for i in range(3):
#     axs[i].plot(Time['train']['Acc'], Dynamics['train']['Acc'][:, i], 'o', label="Training Data", markersize=3)
#     axs[i].plot(Time['valid']['Acc'], Dynamics['valid']['Acc'][:, i], 's', label="Validation Data", markersize=5)
#     axs[i].plot(Time['test']['Acc'],  Dynamics['test']['Acc'][:, i], '.', label="Test Data", markersize=2)
#     axs[i].set_xlabel("Time (s)", fontsize=15)
#     axs[i].set_ylabel(acc_labels[i], fontsize=15)
#     axs[i].set_title(f"Data Partitioning: {acc_labels[i]}", fontsize=18)
#     axs[i].legend(fontsize=12)
#     axs[i].grid(True)

# plt.tight_layout()

# folder_name = f"noise_{int(noise_level*100)}%" if noise_level > 0 else "clean"
# save_dir = os.path.join("data", "VT_DATA", folder_name)
# os.makedirs(save_dir, exist_ok=True)

# plot_file = os.path.join(save_dir, f"{folder_name}_data_plot.png")
# plt.savefig(plot_file, dpi=300, bbox_inches="tight")
# print(f"Data plot saved to: {plot_file}")
# plt.show()


