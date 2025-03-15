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
    def __init__(self, hidden_dim: int = 128):  # hidden_dim를 128로 증가
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
# PART 2: Data Loading
# ============================================================
class DataLoader:
    @staticmethod
    def load_data(noise_level: str, damping: bool = False, uniform: bool = True) -> Tuple[torch.Tensor, ...]:
        base_path = Path("data")
        folder = f"single_pendulum_{'damped' if damping else 'no_damped'}/{noise_level}"
        data_path = base_path / folder

        def load_and_process(prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
            time = np.loadtxt(data_path / f"{noise_level}_input_{prefix}.txt", delimiter='\t', skiprows=1)
            data = np.loadtxt(data_path / f"{noise_level}_output_{prefix}.txt", delimiter='\t', skiprows=1)
            
            time = torch.tensor(time.reshape(-1, 1), dtype=torch.float32)
            data = torch.tensor(data, dtype=torch.float32)
            
            combined = torch.cat((time, data), dim=1)
            sorted_data = combined[combined[:, 0].argsort()]
            return sorted_data[:, 0], sorted_data[:, 1:]

        time_train, data_train = load_and_process("train")
        time_test, data_test = load_and_process("test")
        time_valid, data_valid = load_and_process("valid")

        return time_train, time_test, time_valid, data_train, data_test, data_valid

    @staticmethod
    def get_batch(data: torch.Tensor, time: torch.Tensor, batch_size: int = 32, batch_time: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
def plot_results(time: torch.Tensor, true_data: torch.Tensor, pred_data: torch.Tensor, 
                 title: str, split_idx: int, save_path: Optional[str] = None):
    extrapolation_start_time = time[len(time) // 2]
    plt.figure(figsize=(12, 6))
    plt.plot(time, true_data[:, 2], label='True Data', color='black')
    plt.plot(time[:split_idx], pred_data[:split_idx, 2], label='Interpolation', color='red', linestyle='--', linewidth=2)
    plt.plot(time[split_idx:], pred_data[split_idx:, 2], label='Extrapolation', color='red', linestyle='--', linewidth=2)
    plt.axvline(x=extrapolation_start_time, color='gray',linestyle='--', linewidth=2, label='Extrapolation Start')
    plt.xlabel('Time')
    plt.ylabel('Angular_acceleration')
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
        'noise_level': 0.0,      # 숫자로 입력: 0.0은 clean, 0.4은 noise_40%
        'damping': True,
        'uniform': True,
        'learning_rate': 1e-3,
        'n_iterations': 2000,
        'save_dir': 'example'
    }
    
    # noise_level이 숫자이면 문자열로 변환
    if isinstance(config['noise_level'], (int, float)):
        if config['noise_level'] == 0.0:
            noise_level_str = 'clean'
        else:
            noise_level_str = f"noise_{int(config['noise_level'] * 100)}%"
    else:
        noise_level_str = config['noise_level']
    
    # Load data
    time_train, time_test, time_valid, data_train, data_test, data_valid = DataLoader.load_data(
        noise_level_str, config['damping'], config['uniform']
    )

    # Initialize model and optimizer for training
    # 여기서는 NODE를 사용한 예시지만, 실제로는 NeuralODE와 ODEFunc를 사용합니다.
    node = NeuralODE(ODEFunc())
    best_model_path = f"{config['save_dir']}/trained_models/single_pen_NODE_model_{noise_level_str}.pth"
    optimizer = optim.Adam(node.parameters(), lr=config['learning_rate'], weight_decay=1e-5)  # weight_decay 추가
    trainer = Trainer(node, optimizer, best_model_path)

    training_mode = False  # True: 학습, False: 저장된 모델 불러오기
    if training_mode:
        loss_history = trainer.train(data_train, time_train, data_valid, time_valid, config['n_iterations'])
        
        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.show()
    else:
        node.load_state_dict(torch.load(best_model_path))
        with torch.no_grad():
            pred_y_test = node(data_test[0], time_test, RK4Solver, training=False)
        
        split_idx = 1000 if config['uniform'] else 500
        results_fig_path = f"{config['save_dir']}/figs/single_pen_NODE_fig_{noise_level_str}.png"
        plot_results(time_test, data_test, pred_y_test,
                     f"NODE Single Pendulum {'Uniform' if config['uniform'] else 'Irregular'} Test Data",
                     split_idx,
                     results_fig_path)
        print(f"Result figure saved to {results_fig_path}")

if __name__ == "__main__":
    main()
