import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint

class DoublePendulum:
    """이중 진자 시스템을 시뮬레이션하는 클래스"""
    
    def __init__(self, m1, m2, L1, L2, g, c=0.0, noise_level=0.0):
        """
        Args:
            m1: 첫 번째 진자의 질량 (kg)
            m2: 두 번째 진자의 질량 (kg)
            L1: 첫 번째 진자의 길이 (m)
            L2: 두 번째 진자의 길이 (m)
            g: 중력 가속도 (m/s^2)
            c: 감쇠 계수
            noise_level: 노이즈 수준 (0.0 ~ 1.0)
        """
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g
        self.c = c
        self.noise_level = noise_level
        self.data_type = 'clean' if noise_level == 0.0 else f'noise_{int(noise_level*100)}%'
        
        # 데이터 저장을 위한 속성 초기화
        self.t = None
        self.theta1 = None
        self.omega1 = None
        self.theta2 = None
        self.omega2 = None
        self.alpha1 = None
        self.alpha2 = None
        
        self.theta1_noisy = None
        self.omega1_noisy = None
        self.theta2_noisy = None
        self.omega2_noisy = None
        self.alpha1_noisy = None
        self.alpha2_noisy = None
        
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def ode(self, state, t):
        """이중 진자의 운동 방정식을 정의하는 함수
        
        state = [theta1, omega1, theta2, omega2]
        """
        theta1, omega1, theta2, omega2 = state
        delta = theta1 - theta2
        
        denom = 2*self.m1 + self.m2 - self.m2 * np.cos(2*delta)
        # 각속도 미분 (표준 이중진자 방정식에 간단한 감쇠항 - c*omega를 추가)
        dtheta1 = omega1
        dtheta2 = omega2
        
        domega1 = (
            -self.g*(2*self.m1+self.m2)*np.sin(theta1)
            - self.m2*self.g*np.sin(theta1-2*theta2)
            - 2*np.sin(delta)*self.m2*(omega2**2*self.L2 + omega1**2*self.L1*np.cos(delta))
        ) / (self.L1 * denom) - self.c*omega1
        
        domega2 = (
            2*np.sin(delta)*(omega1**2*self.L1*(self.m1+self.m2)
            + self.g*(self.m1+self.m2)*np.cos(theta1)
            + omega2**2*self.L2*self.m2*np.cos(delta))
        ) / (self.L2 * denom) - self.c*omega2
        
        return [dtheta1, domega1, dtheta2, domega2]

    def generate_data(self, theta1_0, omega1_0, theta2_0, omega2_0, stoptime=10.0, numpoints=5000):
        """주어진 초기 조건으로 이중 진자 운동 데이터를 생성
        
        Args:
            theta1_0: 첫 번째 진자의 초기 각도 (rad)
            omega1_0: 첫 번째 진자의 초기 각속도 (rad/s)
            theta2_0: 두 번째 진자의 초기 각도 (rad)
            omega2_0: 두 번째 진자의 초기 각속도 (rad/s)
            stoptime: 시뮬레이션 시간 (s)
            numpoints: 데이터 포인트 수
        
        Returns:
            tuple: (theta1, omega1, alpha1, theta2, omega2, alpha2) 노이즈 없는 원본 데이터
        """
        self.t = np.linspace(0, stoptime, numpoints)
        initial_conditions = [theta1_0, omega1_0, theta2_0, omega2_0]
        
        solution = odeint(self.ode, initial_conditions, self.t)
        self.theta1 = solution[:, 0]
        self.omega1 = solution[:, 1]
        self.theta2 = solution[:, 2]
        self.omega2 = solution[:, 3]
        
        self.alpha1 = np.gradient(self.omega1, self.t)
        self.alpha2 = np.gradient(self.omega2, self.t)
        
        self._add_noise()
        
        return (self.theta1, self.omega1, self.alpha1,
                self.theta2, self.omega2, self.alpha2)

    def _add_noise(self):
        """생성된 데이터에 가우시안 노이즈 추가"""
        if self.noise_level > 0:
            amps = {
                'theta1': np.max(np.abs(self.theta1)),
                'omega1': np.max(np.abs(self.omega1)),
                'alpha1': np.max(np.abs(self.alpha1)),
                'theta2': np.max(np.abs(self.theta2)),
                'omega2': np.max(np.abs(self.omega2)),
                'alpha2': np.max(np.abs(self.alpha2))
            }
            np.random.seed(42)
            self.theta1_noisy = self.theta1 + np.random.normal(0, self.noise_level * amps['theta1'], self.theta1.shape)
            self.omega1_noisy = self.omega1 + np.random.normal(0, self.noise_level * amps['omega1'], self.omega1.shape)
            self.alpha1_noisy = self.alpha1 + np.random.normal(0, self.noise_level * amps['alpha1'], self.alpha1.shape)
            
            self.theta2_noisy = self.theta2 + np.random.normal(0, self.noise_level * amps['theta2'], self.theta2.shape)
            self.omega2_noisy = self.omega2 + np.random.normal(0, self.noise_level * amps['omega2'], self.omega2.shape)
            self.alpha2_noisy = self.alpha2 + np.random.normal(0, self.noise_level * amps['alpha2'], self.alpha2.shape)
        else:
            self.theta1_noisy = self.theta1.copy()
            self.omega1_noisy = self.omega1.copy()
            self.alpha1_noisy = self.alpha1.copy()
            
            self.theta2_noisy = self.theta2.copy()
            self.omega2_noisy = self.omega2.copy()
            self.alpha2_noisy = self.alpha2.copy()

    def split_data(self, train_size=1000, valid_size=200, test_size=1000, timestep=5):
        """데이터를 학습, 검증, 테스트 세트로 분할
        
        데이터는 시간과 두 진자의 (각도, 각속도, 각가속도)를 포함 (총 7열)
        학습과 검증 데이터는 noisy 데이터를, 테스트 데이터는 clean 데이터를 사용합니다.
        
        Returns:
            tuple: (train_data, valid_data, test_data)
        """
        if self.t is None:
            raise ValueError("먼저 데이터를 생성하세요 (generate_data 메소드 호출)")
        
        # noisy 데이터를 학습/검증용, clean 데이터를 테스트용으로 사용
        data_noisy = np.column_stack((
            self.t,
            self.theta1_noisy,
            self.omega1_noisy,
            self.alpha1_noisy,
            self.theta2_noisy,
            self.omega2_noisy,
            self.alpha2_noisy
        ))
        data_clean = np.column_stack((
            self.t,
            self.theta1,
            self.omega1,
            self.alpha1,
            self.theta2,
            self.omega2,
            self.alpha2
        ))
        
        # 학습 데이터 인덱스 (0부터 시작)
        train_indices = np.arange(0, train_size * timestep, timestep)
        
        # 검증 데이터 인덱스 (offset = 1로 겹치지 않게)
        valid_start = 1
        valid_indices = np.arange(valid_start, valid_start + valid_size * timestep * 5, timestep * 5)
        
        # 테스트 데이터 인덱스 (offset = 2로 겹치지 않게)
        test_start = 2
        test_indices = np.arange(test_start, test_start + test_size * timestep, timestep)
        
        self.train_data = data_noisy[train_indices]
        self.valid_data = data_noisy[valid_indices]
        self.test_data = data_clean[test_indices]
        
        print("\n데이터 분할 정보:")
        print(f"학습 데이터: {len(self.train_data)} 샘플, 시간 범위: [{self.train_data[0,0]:.2f}, {self.train_data[-1,0]:.2f}]")
        print(f"검증 데이터: {len(self.valid_data)} 샘플, 시간 범위: [{self.valid_data[0,0]:.2f}, {self.valid_data[-1,0]:.2f}]")
        print(f"테스트 데이터 (클린): {len(self.test_data)} 샘플, 시간 범위: [{self.test_data[0,0]:.2f}, {self.test_data[-1,0]:.2f}]")
        
        # 인덱스 겹침 확인
        if len(np.intersect1d(train_indices, valid_indices)) == 0:
            print("Train과 Validation 데이터 간 겹침 없음.")
        if len(np.intersect1d(train_indices, test_indices)) == 0:
            print("Train과 Test 데이터 간 겹침 없음.")
        if len(np.intersect1d(valid_indices, test_indices)) == 0:
            print("Validation과 Test 데이터 간 겹침 없음.")
        
        return self.train_data, self.valid_data, self.test_data

    def plot_data(self, save_dir):
        """데이터를 시각화하여 저장
        
        각 데이터셋(Train, Valid, Test)에 대해 한 subplot에 두 진자의
        각도, 각속도, 각가속도를 서로 다른 색상으로 표시.
        """
        if any(x is None for x in [self.train_data, self.valid_data, self.test_data]):
            raise ValueError("먼저 데이터를 분할하세요 (split_data 메소드 호출)")
        
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(15, 15))
        
        # Train 데이터 플롯
        plt.subplot(3, 1, 1)
        plt.scatter(self.train_data[:, 0], self.train_data[:, 1], s=1, color='blue', label='P1 θ')
        plt.scatter(self.train_data[:, 0], self.train_data[:, 2], s=1, color='cyan', label='P1 ω')
        plt.scatter(self.train_data[:, 0], self.train_data[:, 3], s=1, color='navy', label='P1 α')
        plt.scatter(self.train_data[:, 0], self.train_data[:, 4], s=1, color='red', label='P2 θ')
        plt.scatter(self.train_data[:, 0], self.train_data[:, 5], s=1, color='magenta', label='P2 ω')
        plt.scatter(self.train_data[:, 0], self.train_data[:, 6], s=1, color='darkred', label='P2 α')
        plt.title("Train Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Validation 데이터 플롯
        plt.subplot(3, 1, 2)
        plt.scatter(self.valid_data[:, 0], self.valid_data[:, 1], s=1, color='blue', label='P1 θ')
        plt.scatter(self.valid_data[:, 0], self.valid_data[:, 2], s=1, color='cyan', label='P1 ω')
        plt.scatter(self.valid_data[:, 0], self.valid_data[:, 3], s=1, color='navy', label='P1 α')
        plt.scatter(self.valid_data[:, 0], self.valid_data[:, 4], s=1, color='red', label='P2 θ')
        plt.scatter(self.valid_data[:, 0], self.valid_data[:, 5], s=1, color='magenta', label='P2 ω')
        plt.scatter(self.valid_data[:, 0], self.valid_data[:, 6], s=1, color='darkred', label='P2 α')
        plt.title("Validation Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Test 데이터 플롯
        plt.subplot(3, 1, 3)
        plt.scatter(self.test_data[:, 0], self.test_data[:, 1], s=1, color='blue', label='P1 θ')
        plt.scatter(self.test_data[:, 0], self.test_data[:, 2], s=1, color='cyan', label='P1 ω')
        plt.scatter(self.test_data[:, 0], self.test_data[:, 3], s=1, color='navy', label='P1 α')
        plt.scatter(self.test_data[:, 0], self.test_data[:, 4], s=1, color='red', label='P2 θ')
        plt.scatter(self.test_data[:, 0], self.test_data[:, 5], s=1, color='magenta', label='P2 ω')
        plt.scatter(self.test_data[:, 0], self.test_data[:, 6], s=1, color='darkred', label='P2 α')
        plt.title("Test Data (Clean)")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{self.data_type}_data_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_data(self, save_dir):
        """데이터를 텍스트 파일로 저장
        
        Train, Validation, Test 각각에 대해 시간과 두 진자의 (각도, 각속도, 각가속도)를 저장
        """
        if any(x is None for x in [self.train_data, self.valid_data, self.test_data]):
            raise ValueError("먼저 데이터를 분할하세요 (split_data 메소드 호출)")
        
        os.makedirs(save_dir, exist_ok=True)
        datasets = {
            'train': self.train_data,
            'valid': self.valid_data,
            'test': self.test_data
        }
        
        for name, data in datasets.items():
            input_file = os.path.join(save_dir, f'{self.data_type}_input_{name}.txt')
            np.savetxt(input_file, data[:, 0:1], header='time', delimiter='\t', comments='')
            output_file = os.path.join(save_dir, f'{self.data_type}_output_{name}.txt')
            header = 'theta1\tomega1\talpha1\ttheta2\tomega2\talpha2'
            np.savetxt(output_file, data[:, 1:], header=header, delimiter='\t', comments='')
        
        print(f"\n데이터가 {save_dir} 디렉토리에 저장되었습니다.")

# ----------------------------------------------------------------
# 예제 함수: chaotic 및 moderate 상태에 따른 데이터 생성
# ----------------------------------------------------------------
def double_pen_chaotic(noise_level=0.0):
    """Chaotic 상태의 이중 진자 데이터 생성
       - m1=1.0, m2=2.0, L1=2.0, L2=1.0, g=9.81, c는 필요시 설정 (여기서는 0.0)
       - 초기 조건: theta1_0=pi, omega1_0=0, theta2_0=pi/2, omega2_0=0
    """
    noise_level_path = 'clean' if noise_level == 0.0 else f'noise_{int(noise_level*100)}%'
    pendulum = DoublePendulum(m1=1.0, m2=2.0, L1=2.0, L2=1.0, g=9.81, c=0.0, noise_level=noise_level)
    pendulum.generate_data(theta1_0=np.pi, omega1_0=0.0, theta2_0=np.pi/2, omega2_0=0.0, stoptime=10.0, numpoints=10010)
    pendulum.split_data(train_size=1000, valid_size=200, test_size=2000, timestep=5)
    save_dir = os.path.join("data", "double_pendulum", "chaotic", noise_level_path)
    pendulum.plot_data(save_dir=save_dir)
    pendulum.save_data(save_dir=save_dir)

def double_pen_moderate(noise_level=0.0):
    """Moderate 상태의 이중 진자 데이터 생성
       - m1=2.0, m2=0.5, L1=1.0, L2=2.0, g=9.81, c는 필요시 설정 (여기서는 0.0)
       - 초기 조건: theta1_0=1.0, omega1_0=0, theta2_0=0.5, omega2_0=0.3
    """
    noise_level_path = 'clean' if noise_level == 0.0 else f'noise_{int(noise_level*100)}%'
    pendulum = DoublePendulum(m1=2.0, m2=0.5, L1=1.0, L2=2.0, g=9.81, c=0.0, noise_level=noise_level)
    pendulum.generate_data(theta1_0=1.0, omega1_0=0.0, theta2_0=0.5, omega2_0=0.3, stoptime=10.0, numpoints=10010)
    pendulum.split_data(train_size=1000, valid_size=200, test_size=2000, timestep=5)
    save_dir = os.path.join("data", "double_pendulum", "moderate", noise_level_path)
    pendulum.plot_data(save_dir=save_dir)
    pendulum.save_data(save_dir=save_dir)

# ----------------------------------------------------------------
# Main 실행 부분
# ----------------------------------------------------------------
if __name__ == "__main__":
    # 원하는 노이즈 수준 (0.0은 clean, 그 외 0~1 사이 값)
    noise_level = 0.1
    is_chaotic = False  # True: chaotic state, False: moderate state
    
    if is_chaotic:
        print("Chaotic 상태의 이중 진자 데이터 생성:")
        double_pen_chaotic(noise_level=noise_level)
    else:
        print("Moderate 상태의 이중 진자 데이터 생성:")
        double_pen_moderate(noise_level=noise_level)
