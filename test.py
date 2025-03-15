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
# PART 1: ODE Solvers and Augmented NeuralODE Definition
# ============================================================
class RK4Solver:
    @staticmethod
    def solve(func: nn.Module, t: float, dt: float, y: torch.Tensor) -> torch.Tensor:
        """Runge-Kutta 4차 방법으로 ODE를 풉니다."""
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt/2, y + k1/2)
        k3 = dt * func(t + dt/2, y + k2/2)
        k4 = dt * func(t + dt, y + k3)
        return (k1 + 2*k2 + 2*k3 + k4) / 6

class AugmentedNeuralODE(nn.Module):
    def __init__(self, func: nn.Module, input_dim: int = 3, aug_dim: int = 5):
        """
        func: AugmentedODEFunc (입력 차원 + 추가 차원)
        input_dim: 원래 입력 차원 (여기서는 3)
        aug_dim: 추가되는 차원의 개수
        """
        super().__init__()
        self.func = func
        self.input_dim = input_dim
        self.aug_dim = aug_dim

    def forward(self, y0: torch.Tensor, t: torch.Tensor, solver: RK4Solver, training: bool = True) -> torch.Tensor:
        # y0: (batch_size, input_dim)
        # 초기 상태에 추가 차원(aug_dim)을 0으로 채워 확장
        zeros_aug = torch.zeros(y0.shape[0], self.aug_dim, dtype=y0.dtype, device=y0.device)
        y0_aug = torch.cat([y0, zeros_aug], dim=1)  # (batch_size, input_dim + aug_dim)
        
        solution = torch.empty(len(t), *y0_aug.shape, dtype=y0_aug.dtype, device=y0_aug.device)
        solution[0] = y0_aug
        
        if training:
            for i in range(y0_aug.shape[0]):
                y_current = y0_aug[i]
                for j in range(1, len(t)):
                    # t가 2차원 텐서일 경우 각 배치별로 사용
                    if t.ndim == 2:
                        t0, t1 = t[j-1, i], t[j, i]
                    else:
                        t0, t1 = t[j-1], t[j]
                    dy = solver.solve(self.func, t0, t1 - t0, y_current)
                    y_current = y_current + dy
                    solution[j, i] = y_current
        else:
            y_current = y0_aug
            for j in range(1, len(t)):
                t0, t1 = t[j-1], t[j]
                dy = solver.solve(self.func, t0, t1 - t0, y_current)
                y_current = y_current + dy
                solution[j] = y_current
        
        # 최종 결과에서 원래 상태 (첫 input_dim 차원)만 추출
        solution = solution[..., :self.input_dim]
        return solution

class AugmentedODEFunc(nn.Module):
    def __init__(self, input_dim: int = 3, aug_dim: int = 5, hidden_dim: int = 128):
        """
        입력으로 (input_dim + aug_dim) 차원을 받고, 출력도 동일한 차원으로 구성됩니다.
        """
        super().__init__()
        self.input_dim = input_dim
        self.aug_dim = aug_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + aug_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim + aug_dim)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, t: float, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)

# ============================================================
# PART 2: Data Loading for MBD_VT_DATA with Normalization
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoaderMBD:
    @staticmethod
    def load_data(noise_level: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MBD_VT_DATA 데이터 로드 및 전처리 (정규화 포함)
        
        데이터 파일은 폴더 구조: data/MBD_VT_DATA/{noise_level}/
        파일명: {noise_level}_input_{split}.txt, {noise_level}_output_{split}.txt
        
        여기서 input 파일은 시간 데이터만 포함하고, 
        output 파일은 3개의 각가속도 ('Acc_TX@StRq3', 'Acc_TY@StRq3', 'Acc_TZ@StRq3')를 포함합니다.
        
        Returns:
            time_train, time_test, time_valid, data_train, data_test, data_valid
            - time_*: (n,1) 텐서 (시간 데이터)
            - data_*: (n,3) 텐서 (정규화된 각가속도 데이터)
        """
        base_path = Path("data")
        folder = f"MBD_VT_DATA/{noise_level}"
        data_path = base_path / folder

        def load_and_process(prefix: str) -> Tuple[np.ndarray, np.ndarray]:
            # input 파일: 시간 데이터 (1열)
            time = np.loadtxt(data_path / f"{noise_level}_input_{prefix}.txt", delimiter='\t', skiprows=1)
            # output 파일: 3개의 각가속도 데이터
            data = np.loadtxt(data_path / f"{noise_level}_output_{prefix}.txt", delimiter='\t', skiprows=1)
            # 정렬: 시간 기준
            sorted_indices = np.argsort(time)
            time_sorted = time[sorted_indices]
            data_sorted = data[sorted_indices]
            return time_sorted, data_sorted
        
        time_train_np, data_train_np = load_and_process("train")
        time_test_np, data_test_np   = load_and_process("test")
        time_valid_np, data_valid_np = load_and_process("valid")
        
        # 정규화: 학습 데이터 기준으로 StandardScaler 적용 (각가속도 데이터)
        scaler = StandardScaler()
        data_train_np_scaled = scaler.fit_transform(data_train_np)
        data_test_np_scaled  = scaler.transform(data_test_np)
        data_valid_np_scaled = scaler.transform(data_valid_np)
        
        time_train = torch.FloatTensor(time_train_np.reshape(-1, 1))
        time_test = torch.FloatTensor(time_test_np.reshape(-1, 1))
        time_valid = torch.FloatTensor(time_valid_np.reshape(-1, 1))
        data_train = torch.FloatTensor(data_train_np_scaled)
        data_test = torch.FloatTensor(data_test_np_scaled)
        data_valid = torch.FloatTensor(data_valid_np_scaled)
        
        return time_train, time_test, time_valid, data_train, data_test, data_valid

    @staticmethod
    def get_batch(data: torch.Tensor, time: torch.Tensor, batch_size: int = 256, batch_time: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_length = len(data) - batch_time
        start_indices = np.random.choice(data_length, batch_size, replace=False)
        batch_y0 = data[start_indices]
        batch_t = time[:batch_time]
        batch_y = torch.stack([data[idx:idx+batch_time] for idx in start_indices], dim=1)
        return batch_y0, batch_t, batch_y

# ============================================================
# PART 3: Training for Neural ODE (ANODE)
# ============================================================
class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, save_path: str):
        self.model = model
        self.optimizer = optimizer
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
    def train(self, data_train: torch.Tensor, time_train: torch.Tensor, 
              data_valid: torch.Tensor, time_valid: torch.Tensor, 
              n_iterations: int = 2000) -> List[float]:
        best_valid_loss = float('inf')
        loss_history = []
        
        for iter in tqdm(range(n_iterations + 1)):
            self.optimizer.zero_grad()
            batch_y0, batch_t, batch_y = DataLoaderMBD.get_batch(data_train, time_train)
            pred_y = self.model(batch_y0, batch_t, RK4Solver, training=False)
            loss = torch.mean(torch.square(pred_y - batch_y))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            loss_history.append(loss.item())
            
            if iter % 10 == 0:
                with torch.no_grad():
                    pred_y_train = self.model(data_train[0].unsqueeze(0), time_train, RK4Solver, training=False)
                    pred_y_valid = self.model(data_valid[0].unsqueeze(0), time_valid, RK4Solver, training=False)
                    train_loss = torch.mean(torch.abs(pred_y_train - data_train))
                    valid_loss = torch.mean(torch.abs(pred_y_valid - data_valid))
                    print(f'Iter {iter:04d} | Train Loss {train_loss:.6f}, Valid Loss {valid_loss:.6f}')
                    
                    if train_loss < best_valid_loss:  # 보통 valid loss 기준으로 저장합니다.
                        best_valid_loss = train_loss
                        torch.save(self.model.state_dict(), self.save_path)
        return loss_history

# ============================================================
# PART 4: Testing and Plotting for Neural ODE (ANODE)
# ============================================================
def plot_results(time: torch.Tensor, true_data: torch.Tensor, pred_data: torch.Tensor, title: str, save_path: Optional[str] = None):
    """
    결과 시각화 및 저장.
    true_data, pred_data: shape (T, 3)
    여기서는 전체 시간 데이터를 사용하여 3채널 모두 플롯합니다.
    """
    plt.figure(figsize=(12,6))
    channel_labels = ['Acc_TX@StRq3', 'Acc_TY@StRq3', 'Acc_TZ@StRq3']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, true_data[:, i], label=f'True {channel_labels[i]}', color='black', alpha=0.5)
        plt.plot(time, pred_data[:, i], label=f'Predicted {channel_labels[i]}', linestyle='--',
                 color='red' if i==0 else ('blue' if i==1 else 'green'))
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.legend()
        plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

# ============================================================
# PART 5: Main Function for ANODE Training on MBD_VT_DATA
# ============================================================
def main():
    # Configuration
    config = {
        'noise_level': 0.0,   # 0.0: clean, 다른 값은 노이즈 레벨 (예: 0.4는 noise_40%)
        'learning_rate': 1e-4,
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
    
    # MBD_VT_DATA 데이터 로딩 (정규화 포함)
    time_train, time_test, time_valid, data_train, data_test, data_valid = DataLoaderMBD.load_data(noise_level_str)
    
    # 모델과 옵티마이저 초기화: Augmented Neural ODE (ANODE) 모델 사용
    # 입력 차원 3, 추가 차원 5, hidden dimension 128 사용
    anode = AugmentedNeuralODE(
        func=AugmentedODEFunc(input_dim=3, aug_dim=5, hidden_dim=128),
        input_dim=3,
        aug_dim=5
    )
    best_model_path = f"{config['save_dir']}/trained_models/MBD_VT_ANODE_model_{noise_level_str}.pth"
    optimizer = optim.Adam(anode.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    trainer = Trainer(anode, optimizer, best_model_path)
    
    training_mode = True  # True: 학습, False: 저장된 모델 불러오기 및 테스트
    if training_mode:
        loss_history = trainer.train(data_train, time_train, data_valid, time_valid, config['n_iterations'])
        plt.figure(figsize=(10,6))
        plt.plot(loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.show()
    else:
        anode.load_state_dict(torch.load(best_model_path))
        with torch.no_grad():
            # 예측: data_test[0]을 unsqueeze하여 (1, 3) 형태로 전달하면,
            # pred_y_test의 shape는 (len(time_test), 1, 3)이므로 squeeze하여 (len(time_test), 3)으로 변환합니다.
            pred_y_test = anode(data_test[0].unsqueeze(0), time_test, RK4Solver, training=False)
            pred_y_test = pred_y_test.squeeze(1)
            print("Prediction shape:", pred_y_test.shape)
        results_fig_path = f"{config['save_dir']}/figs/MBD_VT_ANODE_fig_{noise_level_str}.png"
        os.makedirs(Path(results_fig_path).parent, exist_ok=True)
        plot_results(time_test, data_test, pred_y_test, f"ANODE MBD_VT Test Data", results_fig_path)
        print(f"Result figure saved to {results_fig_path}")

if __name__ == "__main__":
    main()
