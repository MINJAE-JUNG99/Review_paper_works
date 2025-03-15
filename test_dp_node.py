import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
import logging

# ============================================================
# PART 1: ODE Solvers and NeuralODE Definition
# ============================================================
class RK4Solver:
    @staticmethod
    def solve(func: nn.Module, t: float, dt: float, y: torch.Tensor) -> torch.Tensor:
        """Runge-Kutta 4th order solver."""
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
            # y0 shape: (batch, state_dim)
            for i in range(y0.shape[0]):
                y_current = y0[i]
                for j in range(1, len(t)):
                    t0, t1 = t[j-1], t[j]  # t는 모든 배치에 대해 동일하므로
                    dy = solver.solve(self.func, t0, t1 - t0, y_current)
                    y_current = y_current + dy
                    solution[j, i] = y_current
        else:
            # 비배치 모드: y0가 (state_dim,) 또는 (1, state_dim)
            if y0.dim() == 2:
                y_current = y0[0]
            else:
                y_current = y0
            sol_list = [y_current]
            for j in range(1, len(t)):
                t0, t1 = t[j-1], t[j]
                dy = solver.solve(self.func, t0, t1 - t0, y_current)
                y_current = y_current + dy
                sol_list.append(y_current)
            solution = torch.stack(sol_list, dim=0)
        return solution

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # double pendulum에서 각속도와 각가속도만 사용하므로 입력/출력 차원은 4입니다.
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4)
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
        folder = f"double_pendulum/{'chaotic' if chaotic else 'moderate'}/{noise_level}"
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
            return sorted_data[:, 0], sorted_data[:, [2,3,5,6]]

        time_train, data_train = load_and_process("train")
        time_test, data_test = load_and_process("test")
        time_valid, data_valid = load_and_process("valid")

        return time_train, time_test, time_valid, data_train, data_test, data_valid

    @staticmethod
    def get_batch(data: torch.Tensor, time: torch.Tensor, batch_size: int = 32, batch_time: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_length = len(data) - batch_time
        start_indices = np.random.choice(data_length, batch_size, replace=False)
        
        batch_y0 = data[start_indices]             # (batch_size, state_dim)
        batch_t = time[:batch_time]                # (batch_time,)
        batch_y = torch.stack([data[idx:idx+batch_time] for idx in start_indices], dim=1)  # (batch_time, batch_size, state_dim)
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
            pred_y = self.model(batch_y0, batch_t, RK4Solver, training=True)
            loss = torch.mean((pred_y - batch_y) ** 2)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            loss_history.append(loss.item())
            
            if iter % 10 == 0:
                with torch.no_grad():
                    # 전체 시퀀스 통합: 초기 상태에 대해 전체 시간 예측
                    pred_y_train = self.model(data_train[0].unsqueeze(0), time_train, RK4Solver, training=False).squeeze(0)
                    pred_y_valid = self.model(data_valid[0].unsqueeze(0), time_valid, RK4Solver, training=False).squeeze(0)
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
    double pendulum의 경우, 입력/출력이 4차원입니다.
    여기서는 각가속도만 플롯합니다.
    각가속도: pendulum1 -> index 1, pendulum2 -> index 3
    """
    extrapolation_start_time = time[len(time)//2]
    plt.figure(figsize=(12,6))
    plt.plot(time, true_data[:, 1], label='True Angular Acceleration (Pendulum 1)', color='red', alpha=0.3)
    plt.plot(time, pred_data[:, 1], label='Predicted Angular Acceleration (Pendulum 1)', color='red', linestyle='--', linewidth=2)
    plt.plot(time, true_data[:, 3], label='True Angular Acceleration (Pendulum 2)', color='blue', alpha=0.3)
    plt.plot(time, pred_data[:, 3], label='Predicted Angular Acceleration (Pendulum 2)', color='blue', linestyle='--', linewidth=2)
    plt.axvline(x=extrapolation_start_time, color='gray', linestyle='--', linewidth=2, label='Extrapolation Start')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Acceleration (rad/s²)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

# ============================================================
# PART 5: Main Function
# ============================================================
def main():
    # Configuration
    config = {
        'noise_level': 0.0,   # 0.0: clean, 다른 값은 노이즈 레벨 (예: 0.4는 noise_40%)
        'chaotic': True,     # True이면 chaotic, False이면 moderate
        'uniform': True,
        'learning_rate': 1e-3,
        'n_iterations': 2000,
        'save_dir': 'example'
    }
    
    # noise_level을 문자열 라벨로 변환
    if isinstance(config['noise_level'], (int, float)):
        if config['noise_level'] == 0.0:
            noise_level_str = 'clean'
        else:
            noise_level_str = f"noise_{int(config['noise_level']*100)}%"
    else:
        noise_level_str = config['noise_level']
    
    # double pendulum 데이터 로딩 (입력/출력: angular_velocity와 angular_acceleration, 총 4차원)
    time_train, time_test, time_valid, data_train, data_test, data_valid = DataLoader.load_data(
        noise_level_str, config['chaotic'], config['uniform']
    )
    
    # 모델과 옵티마이저 초기화 (입력/출력 차원 4)
    node = NeuralODE(ODEFunc(hidden_dim=128))
    best_model_path = f"{config['save_dir']}/trained_models/double_pen_NODE_model_{noise_level_str}.pth"
    optimizer = optim.Adam(node.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    trainer = Trainer(node, optimizer, best_model_path)
    
    training_mode = False  # True: 학습, False: 저장된 모델 불러오기 및 테스트
    if training_mode:
        loss_history = trainer.train(data_train, time_train, data_valid, time_valid, config['n_iterations'])
        # 학습 손실 기록 플롯
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
            pred_y_test = node(data_test[0].unsqueeze(0), time_test, RK4Solver, training=False).squeeze(0)
        results_fig_path = f"{config['save_dir']}/figs/double_pen_NODE_fig_{noise_level_str}.png"
        plot_results(time_test, data_test, pred_y_test,
                     f"NODE Double Pendulum {'Uniform' if config['uniform'] else 'Irregular'} Test Data",
                     results_fig_path)
        print(f"Result figure saved to {results_fig_path}")

if __name__ == "__main__":
    main()
