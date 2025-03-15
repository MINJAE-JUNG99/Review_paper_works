import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import logging
from sklearn.preprocessing import StandardScaler
import os
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)  # 예: 'data/MBD_VT_DATA'

    def load_data(self, noise_level: str) -> Tuple[np.ndarray, ...]:
        """
        데이터 로드 및 전처리
        
        Args:
            noise_level: 예를 들어 'clean', 'noise_40%' 등
        
        Returns:
            Tuple: (time_train, time_test, time_valid, data_train, data_test, data_valid)
                   - time_*: (n, 1) 형태의 시간 데이터
                   - data_*: (n, 3) 형태의 각가속도 데이터 (Acc_TX, Acc_TY, Acc_TZ), 정규화 적용
        """
        try:
            file_path = self.base_path / noise_level
            datasets = {}
            for split in ['train', 'test', 'valid']:
                # 입력 파일: 시간 데이터 (n, 1)
                input_data = np.loadtxt(file_path / f"{noise_level}_input_{split}.txt", 
                                         delimiter='\t', skiprows=1)
                # 출력 파일: 각가속도 데이터 (n, 3)
                output_data = np.loadtxt(file_path / f"{noise_level}_output_{split}.txt", 
                                          delimiter='\t', skiprows=1)
                datasets[f"time_{split}"] = input_data.reshape(-1, 1)
                datasets[f"data_{split}"] = output_data
            return self._preprocess_data(datasets)
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise

    def _preprocess_data(self, datasets: dict) -> Tuple[torch.Tensor, ...]:
        """
        데이터 전처리 및 텐서 변환
        
        MBD 데이터는 3열(Acc_TX, Acc_TY, Acc_TZ)로 구성되어 있으며, 
        학습 데이터에서 StandardScaler를 학습하여 검증 및 테스트 데이터에 적용합니다.
        시간 데이터는 정렬 후 그대로 사용합니다.
        """
        processed_data = {}
        
        # 학습 데이터 처리: 시간과 가속도 데이터 결합 및 정렬
        combined_train = np.concatenate((datasets["time_train"], datasets["data_train"]), axis=1)
        sorted_train = combined_train[np.argsort(combined_train[:, 0])]
        
        # 학습 데이터의 각가속도(열 1~3)에 대해 정규화 (시간은 정규화하지 않음)
        scaler = StandardScaler()
        train_acc = sorted_train[:, 1:]
        scaled_train_acc = scaler.fit_transform(train_acc)
        
        processed_data["time_train"] = torch.FloatTensor(sorted_train[:, 0]).unsqueeze(1)
        processed_data["data_train"] = torch.FloatTensor(scaled_train_acc)
        
        # 검증 및 테스트 데이터 처리 (학습 데이터에서 학습한 scaler 사용)
        for split in ['test', 'valid']:
            combined = np.concatenate((datasets[f"time_{split}"], datasets[f"data_{split}"]), axis=1)
            sorted_data = combined[np.argsort(combined[:, 0])]
            acc = sorted_data[:, 1:]
            scaled_acc = scaler.transform(acc)
            
            processed_data[f"time_{split}"] = torch.FloatTensor(sorted_data[:, 0]).unsqueeze(1)
            processed_data[f"data_{split}"] = torch.FloatTensor(scaled_acc)
        
        return (processed_data["time_train"], processed_data["time_test"], 
                processed_data["time_valid"], processed_data["data_train"], 
                processed_data["data_test"], processed_data["data_valid"])

class MLPModel(nn.Module):
    """개선된 MLP 모델 (신경망 두껍게 구성)"""
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

class Trainer:
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

def plot_results(time_test: torch.Tensor, test_data: torch.Tensor, predictions: np.ndarray, 
                 noise_level: str, save_path: str):
    """
    결과 시각화 및 저장 (세 채널 모두 표시)
    
    Args:
        time_test: (n, 1) 텐서 (시간 데이터)
        test_data: (n, 3) 텐서 (실제 각가속도 데이터)
        predictions: (n, 3) numpy 배열 (예측된 각가속도 데이터)
        noise_level: 폴더 이름 지정에 사용되는 문자열 (예: 'clean', 'noise_40%')
        save_path: 플롯 이미지 파일 경로
    """
    time_test = time_test.cpu().numpy().flatten()
    test_data = test_data.cpu().numpy()
    
    plt.figure(figsize=(15, 10))
    colors = ['red', 'blue', 'green']
    labels_true = ['True Acc_TX', 'True Acc_TY', 'True Acc_TZ']
    labels_pred = ['Predicted Acc_TX', 'Predicted Acc_TY', 'Predicted Acc_TZ']
    
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time_test, test_data[:, i], label=labels_true[i], color='black', linestyle='-')
        plt.plot(time_test, predictions[:, i], label=labels_pred[i], color=colors[i], linestyle='--')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Angular Acceleration (rad/s²)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Data plot saved to: {save_path}")

def main():
    # 하이퍼파라미터 설정 (신경망을 두껍게 구성)
    config = {
        'input_size': 1,            # 입력: 시간 t만 사용
        'hidden_sizes': [128, 128, 128, 128],  # 기존 [50, 50]보다 두꺼운 네트워크
        'output_size': 3,           # 출력: 3개의 각가속도 (Acc_TX, Acc_TY, Acc_TZ)
        'num_epochs': 300,          # 에포크 수 증가
        'learning_rate': 0.001,
        'noise_level': 'clean'      # 예: 'clean', 'noise_40%', etc.
    }

    # 데이터 로드 (base_path는 "data/MBD_VT_DATA")
    data_loader = DataLoader('data/MBD_VT_DATA')
    time_train, time_test, time_valid, data_train, data_test, data_valid = \
        data_loader.load_data(config['noise_level'])

    # 모델 초기화 및 학습: 입력은 시간, 타겟은 각가속도 데이터
    model = MLPModel(config['input_size'], config['hidden_sizes'], config['output_size'], dropout_rate=0.1)
    trainer = Trainer(model, config['learning_rate'])

    best_valid_loss = float('inf')
    patience = 50
    patience_counter = 0
    train_losses, valid_losses = [], []
    
    for epoch in range(config['num_epochs']):
        train_loss, valid_loss = trainer.train_epoch(time_train, data_train, time_valid, data_valid)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        trainer.scheduler.step(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            # 최적 모델 저장
            torch.save(model.state_dict(), f'example/trained_models/MBD_VT_FFN_best_model_{config["noise_level"]}.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Train Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}')
    
    # 테스트 데이터에 대해 예측 수행
    predictions = trainer.predict(time_test)
    
    # 결과 플롯 저장: 경로는 예: "example/figs/MBD_VT_FFN_fig_clean.png"
    save_path = f'example/figs/MBD_VT_FFN_fig_{config["noise_level"]}.png'
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plot_results(time_test, data_test, predictions, config['noise_level'], save_path)

if __name__ == '__main__':
    main()
