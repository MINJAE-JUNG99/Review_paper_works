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
# PART 1: ODE Solvers and NeuralODE Definition
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

class NeuralODE(nn.Module):
    def __init__(self, func: nn.Module):
        super().__init__()
        self.func = func

    def forward(self, y0: torch.Tensor, t: torch.Tensor, solver: RK4Solver, training: bool = True) -> torch.Tensor:
        solution = torch.empty(len(t), *y0.shape, dtype=y0.dtype, device=y0.device)
        solution[0] = y0

        if training:
            # Assume y0 has shape (batch, ...)
            for i in range(y0.shape[0]):
                y_current = y0[i]
                for j in range(1, len(t)):
                    t0, t1 = t[j-1, i], t[j, i]
                    dy = solver.solve(self.func, t0, t1 - t0, y_current)
                    y_current = y_current + dy
                    solution[j, i] = y_current
        else:
            y_current = y0
            for j in range(1, len(t)):
                t0, t1 = t[j-1], t[j]
                dy = solver.solve(self.func, t0, t1 - t0, y_current)
                y_current = y_current + dy
                solution[j] = y_current

        return solution

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int = 256):  # hidden_dim를 128로 증가
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t: float, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


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
            # 원래 순서: [time, pos_Tx, vel_Tx, acc_Tx, pos_Ty, vel_Ty, acc_Ty, pos_Tz, vel_Tz, acc_Tz]
            # 여기서는 vel과 acc만 사용: 
            return sorted_data[:, 0], sorted_data[:, [3,6,9]]

        time_train, data_train = load_and_process("train")
        time_test, data_test = load_and_process("test")
        time_valid, data_valid = load_and_process("valid")

        return time_train, time_test, time_valid, data_train, data_test, data_valid

    @staticmethod
    def get_batch(data: torch.Tensor, time: torch.Tensor, batch_size: int = 128, batch_time: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_length = len(data) - batch_time
        start_indices = np.random.choice(data_length, batch_size, replace=False)
        
        batch_y0 = data[start_indices]
        batch_t = time[:batch_time]
        batch_y = torch.stack([data[idx:idx+batch_time] for idx in start_indices], dim=1)
        
        return batch_y0, batch_t, batch_y

# ============================================================
# PART 3: Training
# ============================================================
class Trainer:
    def __init__(self, model: NeuralODE, optimizer: torch.optim.Optimizer, save_path: str):
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
            batch_y0, batch_t, batch_y = DataLoader.get_batch(data_train, time_train)
            pred_y = self.model(batch_y0, batch_t, RK4Solver, training=False)
            loss = torch.mean(torch.square(pred_y - batch_y))
            loss.backward()
            # Gradient clipping 추가
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            loss_history.append(loss.item())
            
            if iter % 10 == 0:
                with torch.no_grad():
                    pred_y_train = self.model(data_train[0], time_train, RK4Solver, training=False)
                    pred_y_valid = self.model(data_valid[0], time_valid, RK4Solver, training=False)
                    train_loss = torch.mean(torch.abs(pred_y_train - data_train))
                    valid_loss = torch.mean(torch.abs(pred_y_valid - data_valid))
                    
                    print(f'Iter {iter:04d} | Train Loss {train_loss:.6f}, Valid Loss {valid_loss:.6f}')
                    
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        torch.save(self.model.state_dict(), self.save_path)
        return loss_history

# ============================================================
# PART 4: Testing and Plotting
# ============================================================
def plot_results(time: torch.Tensor, true_data: torch.Tensor, pred_data: torch.Tensor, title: str, save_path: Optional[str] = None):
    """
    주어진 시간에 대해, true와 predicted 가속도 데이터를 비교하여 플롯합니다.
    여기서는 acc_tx (열 1), acc_ty (열 3), acc_tz (열 5)를 사용합니다.
    
    Args:
        time: 시간 텐서, shape (T, 1) 또는 (T,)
        true_data: True 데이터, shape (T, 6) 
        pred_data: Predicted 데이터, shape (T, 6)
        title: 플롯 제목
        save_path: 저장할 파일 경로 (Optional)
    """
    # 시간 텐서를 numpy 배열로 변환
    time_np = time.numpy().flatten()
    
    # true와 predicted 데이터에서 acc 데이터 추출: acc_tx, acc_ty, acc_tz => 열 1, 3, 5
    true_acc = true_data[:, [1, 3, 5]]
    pred_acc = pred_data[:, [1, 3, 5]]
    
    # extrapolation 시작 시간: 전체 구간의 중간값
    extrapolation_start_time = (time_np[0] + time_np[-1]) / 2
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    acc_labels = ["Acc_TX", "Acc_TY", "Acc_TZ"]
    colors = ['red', 'blue', 'green']
    
    for i in range(3):
        axs[i].plot(time_np, true_acc[:, i], label=f"True {acc_labels[i]}", color=colors[i], alpha=0.3)
        axs[i].plot(time_np, pred_acc[:, i], label=f"Predicted {acc_labels[i]}", color=colors[i], linestyle='--', linewidth=2)
        axs[i].axvline(x=extrapolation_start_time, color='gray', linestyle='--', linewidth=2, label='Extrapolation Start')
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Acceleration (rad/s²)")
        axs[i].set_title(f"{title}: {acc_labels[i]}", fontsize=18)
        axs[i].legend(fontsize=12)
        axs[i].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.show()

# ============================================================
# PART 5: Main Function for Neural ODE Training on MBD_VT_DATA
# ============================================================
def main():
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
    
    # double pendulum 데이터 로딩
    time_train, time_test, time_valid, data_train, data_test, data_valid = DataLoader.load_data(
        noise_level_str, config['chaotic'], config['uniform']
    )
    
    # 모델과 옵티마이저 초기화: NeuralODE 모델 사용
    node = NeuralODE(ODEFunc(hidden_dim=128))
    best_model_path = f"{config['save_dir']}/trained_models/MBD_VT_model_only_acc_{noise_level_str}.pth"
    optimizer = optim.Adam(node.parameters(), lr=config['learning_rate'])
    trainer = Trainer(node, optimizer, best_model_path)
    
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
        node.load_state_dict(torch.load(best_model_path))
        with torch.no_grad():
            # 예측: data_test[0]을 unsqueeze하여 (1, 3) 형태로 전달하면,
            # pred_y_test의 shape는 (len(time_test), 1, 3)이므로 squeeze하여 (len(time_test), 3)으로 변환합니다.
            pred_y_test = node(data_test[0].unsqueeze(0), time_test, RK4Solver, training=False)
            pred_y_test = pred_y_test.squeeze(1)
            print("Prediction shape:", pred_y_test.shape)
        results_fig_path = f"{config['save_dir']}/figs/MBD_VT_model_only_acc_{noise_level_str}.png"
        os.makedirs(Path(results_fig_path).parent, exist_ok=True)
        plot_results(time_test, data_test, pred_y_test, f"NODE MBD_VT Test Data", results_fig_path)
        print(f"Result figure saved to {results_fig_path}")

if __name__ == "__main__":
    main()
