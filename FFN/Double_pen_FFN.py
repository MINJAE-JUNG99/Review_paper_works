import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """데이터 로딩과 전처리를 담당하는 클래스 (double pendulum용)"""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)  # 예: 'data/double_pendulum'

    def load_data(self, noise_level: str, chaotic: bool = False) -> Tuple[np.ndarray, ...]:
        """데이터 로드 및 전처리
        Args:
            noise_level: 예를 들어 'clean', 'noise_40%' 등
            chaotic: True이면 'chaotic' 폴더, False이면 'moderate' 폴더 사용
        """
        try:
            folder = "chaotic" if chaotic else "moderate"
            file_path = self.base_path / folder / noise_level

            # 데이터 로드
            datasets = {}
            for split in ['train', 'test', 'valid']:
                input_data = np.loadtxt(file_path / f"{noise_level}_input_{split}.txt", 
                                         delimiter='\t', skiprows=1)
                output_data = np.loadtxt(file_path / f"{noise_level}_output_{split}.txt", 
                                          delimiter='\t', skiprows=1)
                datasets[f"time_{split}"] = input_data.reshape(-1, 1)
                datasets[f"data_{split}"] = output_data

            return self._preprocess_data(datasets)
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise

    def _preprocess_data(self, datasets: dict) -> Tuple[torch.Tensor, ...]:
        """데이터 전처리 및 텐서 변환  
        double pendulum 데이터는 7열로 구성되어 있습니다:
          [time, theta1, omega1, alpha1, theta2, omega2, alpha2]
        여기서는 오직 각가속도 (alpha1와 alpha2; 인덱스 3와 6)만 선택합니다.
        """
        processed_data = {}
        
        for split in ['train', 'test', 'valid']:
            combined = np.concatenate((datasets[f"time_{split}"], datasets[f"data_{split}"]), axis=1)
            sorted_data = combined[np.argsort(combined[:, 0])]
            
            processed_data[f"time_{split}"] = torch.FloatTensor(sorted_data[:, 0]).unsqueeze(1)
            # 각가속도: alpha1 (인덱스 3)와 alpha2 (인덱스 6); 결과 shape: (n,2)
            processed_data[f"data_{split}"] = torch.FloatTensor(sorted_data[:, [3, 6]])
            
        return (processed_data["time_train"], processed_data["time_test"], 
                processed_data["time_valid"], processed_data["data_train"], 
                processed_data["data_test"], processed_data["data_valid"])

class MLPModel(nn.Module):
    """개선된 MLP 모델"""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 dropout_rate: float = 0.1):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PendulumTrainer:
    """모델 학습과 평가를 관리하는 클래스"""
    def __init__(self, model: nn.Module, learning_rate: float, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, 
                    X_valid: torch.Tensor, y_valid: torch.Tensor) -> Tuple[float, float]:
        self.model.train()
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        self.optimizer.zero_grad()
        y_pred = self.model(X_train)
        train_loss = self.criterion(y_pred, y_train)
        train_loss.backward()
        self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            X_valid, y_valid = X_valid.to(self.device), y_valid.to(self.device)
            y_valid_pred = self.model(X_valid)
            valid_loss = self.criterion(y_valid_pred, y_valid)
            
        return train_loss.item(), valid_loss.item()

    def predict(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            return self.model(X).cpu().numpy()

def plot_results(time_test: np.ndarray, test_data: np.ndarray, predictions: np.ndarray, 
                 noise_level: str, save_path: str):
    """결과 시각화 및 저장 (오직 각가속도만 표시)
       test_data와 predictions는 (n,2) 형태: 
         - 첫 번째 열: Pendulum 1 Angular Acceleration
         - 두 번째 열: Pendulum 2 Angular Acceleration
    """
    plt.figure(figsize=(15, 5))
    
    plt.plot(time_test, test_data[:, 0], label='True P1 Angular Acceleration', color='black', linestyle='-')
    plt.plot(time_test, predictions[:, 0], label='Predicted P1 Angular Acceleration', color='red', linestyle='--')
    
    plt.plot(time_test, test_data[:, 1], label='True P2 Angular Acceleration', color='black', linestyle='-')
    plt.plot(time_test, predictions[:, 1], label='Predicted P2 Angular Acceleration', color='blue', linestyle='--')
    
    # 중앙 시간값을 기준으로 extrapolation 구간 시작선 추가
    extrapolation_start_time = time_test[len(time_test) // 2]
    plt.axvline(x=extrapolation_start_time, color='gray', linestyle='--', linewidth=2, label='Extrapolation Start')
    
    plt.title('Test Data Prediction (Angular Acceleration Only)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Acceleration (rad/s²)')
    plt.legend()
    plt.ylim(-20, 20)  # y축 범위를 -20에서 20으로 설정
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    # 하이퍼파라미터 설정 (double pendulum)
    config = {
        'input_size': 1,            # 입력: 시간 t만 사용
        'hidden_sizes': [50, 50],
        'output_size': 2,           # 출력: 두 진자의 각가속도 (alpha1, alpha2)
        'num_epochs': 200,
        'learning_rate': 0.001,
        'noise_level': 'clean',     # 예: 'clean', 'noise_40%', etc.
        'chaotic': False             # True이면 chaotic, False이면 moderate
    }

    # 데이터 로드 (base_path는 "data/double_pendulum")
    data_loader = DataLoader('data/double_pendulum')
    time_train, time_test, time_valid, data_train, data_test, data_valid = \
        data_loader.load_data(config['noise_level'], config['chaotic'])

    # 모델 초기화 및 학습
    model = MLPModel(config['input_size'], config['hidden_sizes'], config['output_size'], dropout_rate=0.1)
    trainer = PendulumTrainer(model, config['learning_rate'])

    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')
    patience = 50
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        train_loss, valid_loss = trainer.train_epoch(time_train, data_train, time_valid, data_valid)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        trainer.scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'example/trained_models/double_pen_MLP_best_model_{config["noise_level"]}.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Train Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}')

    predictions = trainer.predict(time_test)
    
    plot_results(time_test.numpy(), data_test.numpy(), predictions, 
                 config['noise_level'], f'example/figs/double_pen_MLP_fig_{config["noise_level"]}.png')

if __name__ == "__main__":
    main()
