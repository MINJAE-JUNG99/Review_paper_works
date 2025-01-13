import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint

class SinglePendulum:
    """단일 진자 시스템을 시뮬레이션하는 클래스"""
    
    def __init__(self, m=1.0, L=1.0, g=9.81, c=0.0, noise_level=0.0):
        """
        Args:
            m: 질량 (kg)
            L: 진자 길이 (m)
            g: 중력 가속도 (m/s^2)
            c: 감쇠 계수
            noise_level: 노이즈 수준 (0.0 ~ 1.0)
        """
        self.m = m
        self.L = L
        self.g = g 
        self.c = c
        self.noise_level = noise_level
        self.data_type = 'clean' if noise_level == 0.0 else f'noise_{int(noise_level * 100)}%'
        
        # 데이터 저장을 위한 속성 초기화
        self.t = None
        self.theta = None
        self.omega = None
        self.alpha = None
        self.theta_noisy = None
        self.omega_noisy = None
        self.alpha_noisy = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def ode(self, state, t):
        """진자의 운동 방정식을 정의하는 함수"""
        theta, omega = state
        dydt = [
            omega,
            -(self.g / self.L) * np.sin(theta) - (self.c / self.m) * omega
        ]
        return dydt

    def generate_data(self, theta_0=0.5, omega_0=0.0, stoptime=10.0, numpoints=5000):
        """주어진 초기 조건으로 진자 운동 데이터를 생성
        
        Args:
            theta_0: 초기 각도 (rad)
            omega_0: 초기 각속도 (rad/s)
            stoptime: 시뮬레이션 시간 (s)
            numpoints: 데이터 포인트 수
            
        Returns:
            tuple: (theta, omega, alpha) 노이즈가 없는 원본 데이터
        """
        self.t = np.linspace(0, stoptime, numpoints)
        initial_conditions = [theta_0, omega_0]
        
        solution = odeint(self.ode, initial_conditions, self.t)
        
        self.theta = solution[:, 0]
        self.omega = solution[:, 1]
        self.alpha = np.gradient(self.omega, self.t)
        
        self._add_noise()
        
        return self.theta, self.omega, self.alpha

    def _add_noise(self):
        """생성된 데이터에 가우시안 노이즈 추가"""
        if self.noise_level > 0:
            amplitudes = {
                'theta': np.max(np.abs(self.theta)),
                'omega': np.max(np.abs(self.omega)),
                'alpha': np.max(np.abs(self.alpha))
            }
            
            np.random.seed(42)  # 재현성을 위한 시드 설정
            self.theta_noisy = self.theta + np.random.normal(
                0, self.noise_level * amplitudes['theta'], self.theta.shape)
            self.omega_noisy = self.omega + np.random.normal(
                0, self.noise_level * amplitudes['omega'], self.omega.shape)
            self.alpha_noisy = self.alpha + np.random.normal(
                0, self.noise_level * amplitudes['alpha'], self.alpha.shape)
        else:
            self.theta_noisy = self.theta.copy()
            self.omega_noisy = self.omega.copy()
            self.alpha_noisy = self.alpha.copy()

    def split_data(self, train_size=1000, valid_size=200, test_size=1000, timestep=5):
        """데이터를 학습, 검증, 테스트 세트로 분할
        
        Args:
            train_size: 원하는 학습 데이터 수
            valid_size: 원하는 검증 데이터 수
            test_size: 원하는 테스트 데이터 수
            timestep: 샘플링 간격 (인덱스 간격)
        
        Returns:
            tuple: (train_data, valid_data, test_data) 각각의 데이터셋
        """
        if self.t is None:
            raise ValueError("데이터를 먼저 생성해주세요 (generate_data 메소드 호출)")
            
        # 전체 데이터 배열 생성
        data = np.column_stack((
            self.t,
            self.theta_noisy,
            self.omega_noisy,
            self.alpha_noisy
        ))
        
        
        # 학습 데이터 인덱스 (0부터 시작)
        train_indices = np.arange(0, train_size * timestep, timestep)
        
        # 검증 데이터 인덱스 (offset = 1로 두어서 안 겹치도록)
        valid_start = 1
        valid_indices = np.arange(valid_start, valid_start + valid_size * timestep * 4, timestep*4)
        
        # 테스트 데이터 인덱스 (offset = 2로 두어서 안 겹치도록)
        test_start = 2
        test_indices = np.arange(test_start, test_start + test_size * timestep, timestep)
        
        # 데이터 추출 및 클래스 속성으로 저장
        self.train_data = data[train_indices]
        self.valid_data = data[valid_indices]
        self.test_data = data[test_indices]
        
        # 시간 간격 확인 및 출력
        print("\n데이터 분할 정보:")
        print(f"학습 데이터: {len(self.train_data)} 샘플, 시간 범위: [{self.train_data[0,0]:.2f}, {self.train_data[-1,0]:.2f}]")
        print(f"검증 데이터: {len(self.valid_data)} 샘플, 시간 범위: [{self.valid_data[0,0]:.2f}, {self.valid_data[-1,0]:.2f}]")
        print(f"테스트 데이터: {len(self.test_data)} 샘플, 시간 범위: [{self.test_data[0,0]:.2f}, {self.test_data[-1,0]:.2f}]")
        
        # 겹치는 데이터 확인
        train_val_overlap = np.intersect1d(train_indices, valid_indices)  # Train과 Validation 간의 겹침
        train_test_overlap = np.intersect1d(train_indices, test_indices)  # Train과 Test 간의 겹침
        val_test_overlap = np.intersect1d(valid_indices, test_indices)  # Validation과 Test 간의 겹침
        
        # 결과 출력
        if len(train_val_overlap) == 0:
            print("Train과 Validation 데이터 간에는 겹치는 데이터가 없습니다.")
        else:
            print("Train과 Validation 데이터 간에 겹치는 데이터가 있습니다:", train_val_overlap)
            
        if len(train_test_overlap) == 0:
            print("Train과 Test 데이터 간에는 겹치는 데이터가 없습니다.")
        
        else:
            print("Train과 Test 데이터 간에 겹치는 데이터가 있습니다:", train_test_overlap)
            
        if len(val_test_overlap) == 0:
            print("Validation과 Test 데이터 간에는 겹치는 데이터가 없습니다.")
            
        else:
            print("Validation과 Test 데이터 간에 겹치는 데이터가 있습니다:", val_test_overlap)
            
        return self.train_data, self.valid_data, self.test_data
    
    def plot_data(self, save_dir='data/uniform/single_pendulum_damp'):
        """데이터를 시각화하고 저장
        
        Args:
            save_dir: 그래프를 저장할 디렉토리
        """
        if any(data is None for data in [self.train_data, self.valid_data, self.test_data]):
            raise ValueError("데이터를 먼저 분할해주세요 (split_data 메소드 호출)")
            
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 12))
        
        # 학습 데이터 플롯
        plt.subplot(3, 1, 1)
        plt.scatter(self.train_data[:, 0], self.train_data[:, 1], color='blue', s=1, label='θ')
        plt.scatter(self.train_data[:, 0], self.train_data[:, 2], color='cyan', s=1, label='ω')
        plt.scatter(self.train_data[:, 0], self.train_data[:, 3], color='navy', s=1, label='α')
        plt.title("Train Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        # 검증 데이터 플롯
        plt.subplot(3, 1, 2)
        plt.scatter(self.valid_data[:, 0], self.valid_data[:, 1], color='orange', s=1, label='θ')
        plt.scatter(self.valid_data[:, 0], self.valid_data[:, 2], color='red', s=1, label='ω')
        plt.scatter(self.valid_data[:, 0], self.valid_data[:, 3], color='darkred', s=1, label='α')
        plt.title("Validation Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        # 테스트 데이터 플롯
        plt.subplot(3, 1, 3)
        plt.scatter(self.test_data[:, 0], self.test_data[:, 1], color='green', s=1, label='θ')
        plt.scatter(self.test_data[:, 0], self.test_data[:, 2], color='lime', s=1, label='ω')
        plt.scatter(self.test_data[:, 0], self.test_data[:, 3], color='darkgreen', s=1, label='α')
        plt.title("Test Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{self.data_type}_data_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_data(self, save_dir='data/uniform/single_pendulum_damp'):
        """데이터를 텍스트 파일로 저장
        
        Args:
            save_dir: 데이터를 저장할 디렉토리
        """
        if any(data is None for data in [self.train_data, self.valid_data, self.test_data]):
            raise ValueError("데이터를 먼저 분할해주세요 (split_data 메소드 호출)")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # 데이터셋별로 저장
        datasets = {
            'train': self.train_data,
            'valid': self.valid_data,
            'test': self.test_data
        }
        
        for dataset_name, data in datasets.items():
            # 입력 데이터 저장 (시간)
            input_file = os.path.join(save_dir, f'{self.data_type}_input_{dataset_name}.txt')
            np.savetxt(input_file, data[:, 0:1], 
                      header='time', 
                      delimiter='\t',
                      comments='')
            
            # 출력 데이터 저장 (각도, 각속도, 각가속도)
            output_file = os.path.join(save_dir, f'{self.data_type}_output_{dataset_name}.txt')
            np.savetxt(output_file, data[:, 1:], 
                      header='angle\tangular_velocity\tangular_acceleration',
                      delimiter='\t',
                      comments='')
            
        print(f"\n데이터가 {save_dir} 디렉토리에 저장되었습니다.")


def example_usage():
    """사용 예시"""
    # 진자 시스템 생성
    pendulum = SinglePendulum(m=1.0, L=1.0, g=9.81, c=0.3, noise_level=0.4)
    
    # 데이터 생성
    pendulum.generate_data(theta_0=0.5, omega_0=0.0, stoptime=10.0, numpoints=8010)
    
    # 데이터 분할
    pendulum.split_data(train_size=800, valid_size=200, test_size=1600, timestep=5)
    
    # 데이터 시각화
    pendulum.plot_data()
    
    
    # 데이터 저장
    pendulum.save_data()

if __name__ == "__main__":
    example_usage()