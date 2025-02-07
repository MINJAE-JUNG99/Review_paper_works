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
    """데이터 로딩과 전처리를 담당하는 클래스"""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def load_data(self, noise_level: str, damping: bool = False) -> Tuple[np.ndarray, ...]:
        """데이터 로드 및 전처리"""
        try:
            folder = "single_pendulum_damped" if damping else "single_pendulum_undamped"
            file_path = self.base_path / "test" / folder / noise_level

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
        """데이터 전처리 및 텐서 변환"""
        processed_data = {}
        
        for split in ['train', 'test', 'valid']:
            # 시간과 데이터 결합 및 정렬
            combined = np.concatenate((datasets[f"time_{split}"], 
                                     datasets[f"data_{split}"]), axis=1)
            sorted_data = combined[np.argsort(combined[:, 0])]
            
            # 시간과 데이터 분리
            processed_data[f"time_{split}"] = torch.FloatTensor(sorted_data[:, 0]).unsqueeze(1)
            processed_data[f"data_{split}"] = torch.FloatTensor(sorted_data[:, 1:])

        return (processed_data["time_train"], processed_data["time_test"], 
                processed_data["time_valid"], processed_data["data_train"], 
                processed_data["data_test"], processed_data["data_valid"])

class MLPModel(nn.Module):
    """개선된 MLP 모델"""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 dropout_rate: float = 0.1):
        super(MLPModel, self).__init__()
        
        # 동적으로 레이어 생성
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                  mode='min', 
                                                                  factor=0.5, 
                                                                  patience=20, 
                                                                  verbose=True)

    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, 
                   X_valid: torch.Tensor, y_valid: torch.Tensor) -> Tuple[float, float]:
        """한 에폭 학습 및 검증"""
        # 학습
        self.model.train()
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        
        self.optimizer.zero_grad()
        y_pred = self.model(X_train)
        train_loss = self.criterion(y_pred, y_train)
        train_loss.backward()
        self.optimizer.step()

        # 검증
        self.model.eval()
        with torch.no_grad():
            X_valid, y_valid = X_valid.to(self.device), y_valid.to(self.device)
            y_valid_pred = self.model(X_valid)
            valid_loss = self.criterion(y_valid_pred, y_valid)
            
        return train_loss.item(), valid_loss.item()

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """테스트 데이터 예측"""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            return self.model(X).cpu().numpy()

def plot_results(time_test: np.ndarray, test_data: np.ndarray, predictions: np.ndarray, 
                 noise_level: str, save_path: str):
    """결과 시각화 및 저장 (수정된 버전)"""
    plt.figure(figsize=(15, 5))

    # 색상 및 레이블 설정
    colors = ['red', 'blue', 'green']
    labels = ['Angle', 'Angular Velocity', 'Angular Acceleration']

    # Extrapolation 구간 시작 시간 설정 (중앙 5초 구간)
    extrapolation_start_time = time_test[len(time_test) // 2]  # 중앙 시간값

    for i in range(3):
        plt.plot(time_test, test_data[:, i], label=f'True {labels[i]}', 
                 color='black', linestyle='-')
        plt.plot(time_test, predictions[:, i], label=f'Predicted {labels[i]}', 
                 color=colors[i], linestyle='--')

    # Extrapolation 구간 시작점에 수직선 추가
    plt.axvline(x=extrapolation_start_time, color='gray', linestyle='--', linewidth=2, 
                label='Extrapolation Start')

    plt.title('Test Data Prediction')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    # 하이퍼파라미터 설정
    config = {
        'input_size': 1,
        'hidden_sizes': [50, 50],
        'output_size': 3,
        'num_epochs': 200,
        'learning_rate': 0.001,
        'noise_level': 'clean',
        'damping': True
    }

    # 데이터 로드
    data_loader = DataLoader('data')
    time_train, time_test, time_valid, data_train, data_test, data_valid = \
        data_loader.load_data(config['noise_level'], config['damping'])

    # 모델 초기화 및 학습
    model = MLPModel(config['input_size'], config['hidden_sizes'], 
                    config['output_size'], dropout_rate=0.1)
    trainer = PendulumTrainer(model, config['learning_rate'])

    # 학습 루프
    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')
    patience = 50
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        train_loss, valid_loss = trainer.train_epoch(time_train, data_train, 
                                                   time_valid, data_valid)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Learning rate 조정
        trainer.scheduler.step(valid_loss)

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            # 최상의 모델 저장
            torch.save(model.state_dict(), f'MLP/best_model_{config["noise_level"]}.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch + 1}/{config["num_epochs"]}], '
                       f'Train Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}')

    # 테스트 데이터로 예측
    predictions = trainer.predict(time_test)
    
    # 결과 플롯 및 저장
    plot_results(time_test.numpy(), data_test.numpy(), predictions, 
                config['noise_level'], f'MLP/MLP_results_time_only_{config["noise_level"]}.png')

if __name__ == "__main__":
    main()