import numpy as np
import matplotlib.pyplot as plt
import os

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

def normalize_data(data, method="z-score"):
    if method == "z-score":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / (std + 1e-8)
    elif method == "min-max":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError("method는 'z-score' 또는 'min-max' 중 하나여야 합니다.")
    return normalized_data

#########################################
# 1. CSV 파일 로드 및 데이터 전처리
#########################################
# noise_level_value 설정
# 여기서는 학습/검증 데이터에만 노이즈(예: 10%)를 적용하고, 테스트 데이터는 clean 데이터를 사용합니다.
noise_level = 0.1  

# CSV 파일 경로 (파일 이름과 경로는 실제 환경에 맞게 조정)
file_path_acc = "data/MBD_VT_DATA/VT_Dense_PointA_Acc.csv"
file_path_vel = "data/MBD_VT_DATA/VT_Dense_PointA_Vel.csv"
file_path_pos = "data/MBD_VT_DATA/VT_Dense_PointA_Pos.csv"

# CSV 파일 로드
acc_full = np.genfromtxt(file_path_acc, delimiter=",", skip_header=1)
vel_full = np.genfromtxt(file_path_vel, delimiter=",", skip_header=1)
pos_full = np.genfromtxt(file_path_pos, delimiter=",", skip_header=1)

acc_full = acc_full[:, 2:]  # No, TIME 열 제외
vel_full = vel_full[:, 2:]  # No, TIME 열 제외
pos_full = pos_full[:, 2:]  # No, TIME 열 제외

# 전체 데이터 포인트 수 (여기서는 예시로 25000개 사용)
n_samples = 25000  
time_all = np.linspace(0, 10, n_samples, endpoint=True)
# 만약 CSV의 TIME 열을 그대로 사용하고 싶다면 주석 해제:
# time_all = acc_full[:, 1]

# 정규화 (clean 데이터 기준)
acc_data_clean = normalize_data(acc_full, method="z-score")
vel_data_clean = normalize_data(vel_full, method="z-score")
pos_data_clean = normalize_data(pos_full, method="z-score")

# 학습/검증 데이터에는 노이즈 추가 (noise_level_value > 0 인 경우)
if noise_level > 0:
    acc_data_noisy = add_noise(acc_data_clean, noise_level=noise_level, seed=42)
    vel_data_noisy = add_noise(vel_data_clean, noise_level=noise_level, seed=42)
    pos_data_noisy = add_noise(pos_data_clean, noise_level=noise_level, seed=42)
else:
    acc_data_noisy = acc_data_clean.copy()
    vel_data_noisy = vel_data_clean.copy()
    pos_data_noisy = pos_data_clean.copy()

n_total = time_all.shape[0]

#########################################
# 2. 데이터셋 분할 (균등한 선택)
#########################################
# 여기서는 인덱스 생성에 np.arange를 사용합니다.
# 학습/검증 데이터는 전체 데이터의 앞부분을 사용하고, 테스트 데이터는 전체 데이터에서 학습/검증에 사용되지 않은 인덱스를 사용합니다.

timestep = 5        # 인덱스 간격
train_size = 2500   # 학습 데이터 샘플 개수
valid_size = 500    # 검증 데이터 샘플 개수
test_size  = 4999   # 테스트 데이터 샘플 개수

# 학습 데이터 인덱스: 0부터 시작, 일정한 timestep 간격
train_indices = np.arange(0, train_size * timestep, timestep)

# 검증 데이터 인덱스: 학습 데이터와 겹치지 않도록 offset=1 사용, 간격은 timestep*10 (예시)
valid_start = 1
valid_indices = np.arange(valid_start, valid_start + valid_size * timestep * 5, timestep * 5)

# 테스트 데이터 인덱스: 전체 데이터에서 학습 및 검증에 사용되지 않은 인덱스 중 offset=2 사용
test_start = 2
test_indices = np.arange(test_start, test_start + test_size * timestep, timestep)

#########################################
# 3. 데이터셋 구성
#########################################
# 학습 및 검증 데이터: 노이즈가 적용된 데이터 사용
train_time = time_all[train_indices]
train_pos = pos_data_noisy[train_indices]
train_vel = vel_data_noisy[train_indices]
train_acc = acc_data_noisy[train_indices]

valid_time = time_all[valid_indices]
valid_pos = pos_data_noisy[valid_indices]
valid_vel = vel_data_noisy[valid_indices]
valid_acc = acc_data_noisy[valid_indices]

# 테스트 데이터: clean 데이터 사용 (노이즈 없음)
test_time = time_all[test_indices]
test_pos = pos_data_clean[test_indices]
test_vel = vel_data_clean[test_indices]
test_acc = acc_data_clean[test_indices]

# 데이터셋 간 겹침 여부 확인
intersect_train_val = np.intersect1d(train_indices, valid_indices)
intersect_train_test = np.intersect1d(train_indices, test_indices)
intersect_val_test = np.intersect1d(valid_indices, test_indices)

print("Intersection between training and validation indices:", intersect_train_val)
print("Intersection between training and test indices:", intersect_train_test)
print("Intersection between validation and test indices:", intersect_val_test)
#########################################
# 4. 데이터셋 플롯 (세 축 모두 플롯)
#########################################
def plot_dataset(time_data, pos_data, vel_data, acc_data, dataset_name, save_path=None):
    labels = ["TX@StRq3", "TY@StRq3", "TZ@StRq3"]
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    for i in range(3):
        axs[i].plot(time_data, pos_data[:, i], 'o', label="Pos", markersize=3)
        axs[i].plot(time_data, vel_data[:, i], 's', label="Vel", markersize=3)
        axs[i].plot(time_data, acc_data[:, i], '.', label="Acc", markersize=2)
        axs[i].set_xlabel("Time (s)", fontsize=15)
        axs[i].set_ylabel(labels[i], fontsize=15)
        axs[i].set_title(f"{dataset_name} Data: {labels[i]}", fontsize=18)
        axs[i].legend(fontsize=12)
        axs[i].grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"{dataset_name} plot saved to: {save_path}")
    # plt.show()


# 예시: 데이터셋(Train, Valid, Test) 별 플롯을 데이터 저장 경로에 저장
folder_name = f"noise_{int(noise_level*100)}%" if noise_level > 0 else "clean"
base_save_dir = os.path.join("data", "MBD_VT_DATA", folder_name)
os.makedirs(base_save_dir, exist_ok=True)

# 플롯 저장 파일 경로
train_plot_path = os.path.join(base_save_dir, "Train_data_plot.png")
valid_plot_path = os.path.join(base_save_dir, "Valid_data_plot.png")
test_plot_path  = os.path.join(base_save_dir, "Test_data_plot.png")

# 각각의 데이터셋 플롯 저장
plot_dataset(train_time, train_pos, train_vel, train_acc, "Train", save_path=train_plot_path)
plot_dataset(valid_time, valid_pos, valid_vel, valid_acc, "Valid", save_path=valid_plot_path)
plot_dataset(test_time, test_pos, test_vel, test_acc, "Test", save_path=test_plot_path)

# #########################################
# # 5. 데이터 저장 함수: 텍스트 파일로 저장
# #########################################
def save_partition(time_data, pos_data, vel_data, acc_data, partition, folder_name, base_dir):
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    # 시간 데이터 저장 (2차원 배열로 reshape)
    input_file = os.path.join(save_dir, f"{folder_name}_input_{partition}.txt")
    np.savetxt(input_file, time_data.reshape(-1, 1), delimiter='\t', header="Time", comments="")

    # X, Y, Z 축별로 pos → vel → acc 순서로 배열 결합
    combined = np.column_stack((
        pos_data[:, 0], vel_data[:, 0], acc_data[:, 0],  # X축
        pos_data[:, 1], vel_data[:, 1], acc_data[:, 1],  # Y축
        pos_data[:, 2], vel_data[:, 2], acc_data[:, 2]   # Z축
    ))

    # 올바른 헤더 설정
    output_file = os.path.join(save_dir, f"{folder_name}_output_{partition}.txt")
    header = "\t".join([
        "Pos_TX", "Vel_TX", "Acc_TX",
        "Pos_TY", "Vel_TY", "Acc_TY",
        "Pos_TZ", "Vel_TZ", "Acc_TZ"
    ])
    
    np.savetxt(output_file, combined, delimiter='\t', header=header, comments="")
    print(f"Saved {partition} dataset to {output_file}")

# 저장: 학습, 검증, 테스트 데이터 각각 저장
base_save_dir = "data/MBD_VT_DATA"
save_partition(train_time, train_pos, train_vel, train_acc, "train", folder_name, base_save_dir)
save_partition(valid_time, valid_pos, valid_vel, valid_acc, "valid", folder_name, base_save_dir)
save_partition(test_time, test_pos, test_vel, test_acc, "test", folder_name, base_save_dir)
