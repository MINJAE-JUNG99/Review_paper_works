import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional

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
    def __init__(self, hidden_dim: int = 64):
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

class DataLoader:
    @staticmethod
    def load_data(noise_level: str, damping: bool = False, uniform: bool = True) -> Tuple[torch.Tensor, ...]:
        base_path = Path("data/test")
        folder = f"{'uniform/' if uniform else ''}single_pendulum_{'damped' if damping else 'undamped'}/{noise_level}"
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
            # Training step
            self.optimizer.zero_grad()
            batch_y0, batch_t, batch_y = DataLoader.get_batch(data_train, time_train)
            pred_y = self.model(batch_y0, batch_t, RK4Solver, training=False)
            loss = torch.mean(torch.square(pred_y - batch_y))
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())

            # Validation step
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

def plot_results(time: torch.Tensor, true_data: torch.Tensor, pred_data: torch.Tensor, 
                title: str, split_idx: int, save_path: Optional[str] = None):
    plt.figure(figsize=(12, 6))
    plt.plot(time, true_data[:, 0], label='True Data', color='black', alpha=0.3)
    plt.plot(time[:split_idx], pred_data[:split_idx, 0], label='Interpolation', color='red', linewidth=2)
    plt.plot(time[split_idx:], pred_data[split_idx:, 0], label='Extrapolation', color='magenta', linestyle='--', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    config = {
        'noise_level': 'clean',
        'damping': True,
        'uniform': True,
        'learning_rate': 1e-3,
        'n_iterations': 2000,
        'save_dir': 'NODE/runs_uniform'
    }

    # Load data
    time_train, time_test, time_valid, data_train, data_test, data_valid = DataLoader.load_data(
        config['noise_level'], config['damping'], config['uniform']
    )

    # Initialize model and optimizer
    node = NeuralODE(ODEFunc())
    optimizer = optim.Adam(node.parameters(), lr=config['learning_rate'])
    trainer = Trainer(node, optimizer, f"{config['save_dir']}/best_model.pt")

    # Training or testing
    training_mode = False
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
        # Load best model and test
        node.load_state_dict(torch.load(f"{config['save_dir']}/best_model.pt"))
        with torch.no_grad():
            pred_y_test = node(data_test[0], time_test, RK4Solver, training=False)
        
        split_idx = 800 if config['uniform'] else 500
        plot_results(
            time_test, data_test, pred_y_test,
            f"NODE Single Pendulum {'Uniform' if config['uniform'] else 'Irregular'} Test Data",
            split_idx,
            f"{config['save_dir']}/test_results.png"
        )

if __name__ == "__main__":
    main()