import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class MLP(nn.Module):
    """단일 진자 예측을 위한 MLP 모델"""
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

def load_data(input_file, output_file):
    """텍스트 파일에서 데이터를 로드하고 정규화"""
    # 데이터 로드
    input_data = np.loadtxt(input_file, delimiter='\t', skiprows=1)
    output_data = np.loadtxt(output_file, delimiter='\t', skiprows=1)

    # 정규화
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    input_data = scaler_x.fit_transform(input_data)
    output_data = scaler_y.fit_transform(output_data[:, 0:1])  # θ만 예측 대상으로 사용

    return input_data, output_data, scaler_x, scaler_y

def train_mlp_model(input_data, output_data, epochs=1000, learning_rate=0.001, batch_size=32):
    """MLP 모델 학습"""
    # 데이터를 PyTorch 텐서로 변환
    input_data = torch.tensor(input_data, dtype=torch.float32)
    output_data = torch.tensor(output_data, dtype=torch.float32)

    # 데이터셋 분할
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    # DataLoader 생성
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    model = MLP(input_size=x_train.shape[1], hidden_size=64, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.6f}")

    # 테스트 데이터에서 모델 평가
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        test_loss = criterion(predictions, y_test).item()
        print(f"Test Loss: {test_loss:.6f}")

    return model

def predict(model, input_data, scaler_y):
    """모델로 예측 수행"""
    model.eval()
    input_data = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(input_data)
    predictions = predictions.numpy()
    predictions = scaler_y.inverse_transform(predictions)
    return predictions

if __name__ == "__main__":
    # 데이터 파일 경로
    input_file = 'Data/single_pendulum_undamped/clean_input_train.txt'
    output_file = 'Data/single_pendulum_undamped/clean_output_train.txt'

    # 데이터 로드
    input_data, output_data, scaler_x, scaler_y = load_data(input_file, output_file)

    # 모델 학습
    model = train_mlp_model(input_data, output_data, epochs=1000, learning_rate=0.001, batch_size=32)

    # 테스트 예측
    test_input_file = 'Data/single_pendulum_undamped/clean_input_test.txt'
    test_output_file = 'Data/single_pendulum_undamped/clean_output_test.txt'

    test_input, _, _, _ = load_data(test_input_file, test_output_file)
    predictions = predict(model, test_input, scaler_y)

    # 예측 결과 확인
    print("Sample Predictions:", predictions[:10])
