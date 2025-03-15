import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
import logging
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============================================================
# PART 2: Data Loading for Double Pendulum
# ============================================================
class DataLoader:
    @staticmethod
    def load_data(noise_level: str, chaotic: bool = True, uniform: bool = True) -> Tuple[torch.Tensor, ...]:
        base_path = Path("data")
        folder = f"MBD_VT_DATA/{noise_level}"
        data_path = base_path / folder

        def load_and_process(prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
            time = np.loadtxt(data_path / f"{noise_level}_input_{prefix}.txt", delimiter='\t', skiprows=1)
            data = np.loadtxt(data_path / f"{noise_level}_output_{prefix}.txt", delimiter='\t', skiprows=1)
            
            time = torch.tensor(time.reshape(-1, 1), dtype=torch.float32)
            data = torch.tensor(data, dtype=torch.float32)
            
            combined = torch.cat((time, data), dim=1)
            sorted_data = combined[combined[:, 0].argsort()]
            # 원래 순서: [time, angle1, angular_velocity1, angular_acceleration1, angle2, angular_velocity2, angular_acceleration2]
            # 여기서는 angular_velocity와 angular_acceleration만 사용: pendulum1: columns 2,3 / pendulum2: columns 5,6
            return sorted_data[:, 0], sorted_data[:, [1,2,3]]

        time_train, data_train = load_and_process("train")
        time_test, data_test = load_and_process("test")
        time_valid, data_valid = load_and_process("valid")

        return time_train, time_test, time_valid, data_train, data_test, data_valid
    
# Configuration
config = {
        'noise_level': 0.0,   # 0.0: clean, 다른 값은 노이즈 레벨 (예: 0.4는 noise_40%)
        'chaotic': False,      # True이면 chaotic, False이면 moderate
        'uniform': True,
        'learning_rate': 1e-3,
        'n_iterations': 2000,
        'save_dir': 'example'
}
    
# noise_level 문자열 변환
if isinstance(config['noise_level'], (int, float)):
    if config['noise_level'] == 0.0:
            noise_level_str = 'clean'
    else:
            noise_level_str = f"noise_{int(config['noise_level']*100)}%"
else:
        noise_level_str = config['noise_level']
        
        
time_train, time_test, time_valid, data_train, data_test, data_valid = DataLoader.load_data(
        noise_level_str, config['chaotic'], config['uniform']
)

def plot_data(time: torch.Tensor, data: torch.Tensor, dataset_name: str):
    # 텐서를 numpy 배열로 변환
    time_np = time.numpy().flatten()
    data_np = data.numpy()
    n_channels = data_np.shape[1]
    
    # 데이터 절반에 해당하는 시간 값 계산
    half_time = time_np[len(time_np) // 2]
    
    fig, axs = plt.subplots(n_channels, 1, figsize=(10, 3 * n_channels))
    for i in range(n_channels):
        axs[i].plot(time_np, data_np[:, i], label=f"Channel {i+1}", marker='o', markersize=3)
        # 중간 시간에 수직선 추가
        axs[i].axvline(x=half_time, color='gray', linestyle='--', linewidth=2, label='Half Point')
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Value")
        axs[i].set_title(f"{dataset_name} - Channel {i+1}")
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================================================
# 각 데이터셋에 대해 플롯 그리기
# ============================================================
plot_data(time_test, data_test, "Test Data")
print("Number of test data samples:", time_test.shape[0])