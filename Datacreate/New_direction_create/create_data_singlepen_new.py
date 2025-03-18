import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def get_paths(config, makedirs=True, folders=['data']):
    """
    config를 받아 single_pendulum 경로를 생성.
    예: './data/single_pendulum/damped0.3_clean/' 형태로 디렉토리 생성.
    """
    paths = {}
    
    example = config['example']        # 예: 'single_pendulum'
    damping = config['damping_coef']   # 예: 0.3
    noise_str = config['noise_level']  # 예: 'clean', 'noise40%'
    
    # 예: 'damped0.3_clean', 'damped0.3_noise40%' 등
    subfolder = f"damped{damping}_{noise_str}"
    
    for folder in folders:
        # 예: './data/single_pendulum/damped0.3_clean/'
        path = os.path.join('.', folder, example, subfolder)
        if makedirs:
            os.makedirs(path, exist_ok=True)
        paths[folder] = path
    
    return paths

class SinglePendulum:
    """단일 진자 시스템 시뮬레이션 클래스"""
    
    def __init__(self, m=1.0, L=1.0, g=9.81, c=0.0, noise_level=0.0):
        self.m = m
        self.L = L
        self.g = g
        self.c = c
        
        # 노이즈 레벨에 따라 clean / noiseXX% 문자열만 관리 (파일명에는 쓰지 않음)
        if noise_level == 0.0:
            self.data_type = 'clean'
        else:
            self.data_type = f'noise_{int(noise_level*100)}%'
        
        self.noise_level = noise_level
        
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
        """단일 진자 운동 방정식"""
        theta, omega = state
        dtheta_dt = omega
        domega_dt = -(self.g / self.L) * np.sin(theta) - (self.c / self.m) * omega
        return [dtheta_dt, domega_dt]
    
    def generate_data(self, theta_0=0.5, omega_0=0.0, stoptime=10.0, numpoints=5000):
        """주어진 초기 조건으로 진자 운동 데이터를 생성"""
        self.t = np.linspace(0, stoptime, numpoints)
        sol = odeint(self.ode, [theta_0, omega_0], self.t)
        self.theta = sol[:, 0]
        self.omega = sol[:, 1]
        self.alpha = np.gradient(self.omega, self.t)
        self._add_noise()
        return self.theta, self.omega, self.alpha

    def _add_noise(self):
        """노이즈 추가 (노이즈 레벨이 0이면 clean 그대로)"""
        if self.noise_level > 0.0:
            np.random.seed(42)
            amp_theta = np.max(np.abs(self.theta))
            amp_omega = np.max(np.abs(self.omega))
            amp_alpha = np.max(np.abs(self.alpha))
            
            self.theta_noisy = self.theta + np.random.normal(0, self.noise_level * amp_theta, size=self.theta.shape)
            self.omega_noisy = self.omega + np.random.normal(0, self.noise_level * amp_omega, size=self.omega.shape)
            self.alpha_noisy = self.alpha + np.random.normal(0, self.noise_level * amp_alpha, size=self.alpha.shape)
        else:
            self.theta_noisy = self.theta.copy()
            self.omega_noisy = self.omega.copy()
            self.alpha_noisy = self.alpha.copy()

    def split_data(self, train_size=1000, valid_size=200, test_size=1000, timestep=5):
        """데이터를 학습, 검증, 테스트 세트로 분할"""
        if self.t is None:
            raise ValueError("generate_data() 먼저 호출해주세요.")
        
        # noisy -> train, valid / clean -> test
        data_noisy = np.column_stack((self.t, self.theta_noisy, self.omega_noisy, self.alpha_noisy))
        data_clean = np.column_stack((self.t, self.theta, self.omega, self.alpha))
        
        train_indices = np.arange(0, train_size * timestep, timestep)
        valid_indices = np.arange(1, 1 + valid_size * timestep * 5, timestep*5)
        test_indices  = np.arange(2, 2 + test_size  * timestep, timestep)
        
        self.train_data = data_noisy[train_indices]
        self.valid_data = data_noisy[valid_indices]
        self.test_data  = data_clean[test_indices]
        
        print("[Data Split]")
        print(f"  Train: {len(self.train_data)} samples, time range [{self.train_data[0,0]:.2f}, {self.train_data[-1,0]:.2f}]")
        print(f"  Valid: {len(self.valid_data)} samples, time range [{self.valid_data[0,0]:.2f}, {self.valid_data[-1,0]:.2f}]")
        print(f"  Test (clean): {len(self.test_data)} samples, time range [{self.test_data[0,0]:.2f}, {self.test_data[-1,0]:.2f}]")
        
        return self.train_data, self.valid_data, self.test_data

    def plot_data(self, save_dir):
        """
        학습/검증/테스트 데이터를 동일 폴더(save_dir)에 data_plot.png로 저장.
        """
        if any(d is None for d in [self.train_data, self.valid_data, self.test_data]):
            raise ValueError("split_data() 먼저 호출해주세요.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        
        # Train
        axs[0].scatter(self.train_data[:, 0], self.train_data[:, 1], s=2, label='θ', color='blue')
        axs[0].scatter(self.train_data[:, 0], self.train_data[:, 2], s=2, label='ω', color='cyan')
        axs[0].scatter(self.train_data[:, 0], self.train_data[:, 3], s=2, label='α', color='navy')
        axs[0].set_title("Train Data")
        axs[0].legend()
        axs[0].grid(True)
        
        # Valid
        axs[1].scatter(self.valid_data[:, 0], self.valid_data[:, 1], s=2, label='θ', color='orange')
        axs[1].scatter(self.valid_data[:, 0], self.valid_data[:, 2], s=2, label='ω', color='red')
        axs[1].scatter(self.valid_data[:, 0], self.valid_data[:, 3], s=2, label='α', color='darkred')
        axs[1].set_title("Validation Data")
        axs[1].legend()
        axs[1].grid(True)
        
        # Test (clean)
        axs[2].scatter(self.test_data[:, 0], self.test_data[:, 1], s=2, label='θ', color='green')
        axs[2].scatter(self.test_data[:, 0], self.test_data[:, 2], s=2, label='ω', color='lime')
        axs[2].scatter(self.test_data[:, 0], self.test_data[:, 3], s=2, label='α', color='darkgreen')
        axs[2].set_title("Test Data (Clean)")
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "data_plot.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Plot saved: {plot_path}")

    def save_data(self, save_dir):

        if any(d is None for d in [self.train_data, self.valid_data, self.test_data]):
            raise ValueError("split_data() 먼저 호출해주세요.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 각 데이터셋을 (time) / (theta, omega, alpha)로 나누어 저장
        dataset_map = {
            'train': self.train_data,
            'valid': self.valid_data,
            'test':  self.test_data
        }
        
        for name, data in dataset_map.items():
            # 입력 (시간)
            in_file = os.path.join(save_dir, f"input_{name}.txt")
            np.savetxt(in_file, data[:, [0]], header='time', delimiter='\t', comments='')
            
            # 출력 (θ, ω, α)
            out_file = os.path.join(save_dir, f"output_{name}.txt")
            np.savetxt(out_file, data[:, 1:], 
                       header='angle\tangular_velocity\tangular_acceleration',
                       delimiter='\t', comments='')
        
        print(f"Data (txt) saved under: {save_dir}")

def single_pen_damped(noise_level=0.0, damping_coeff=0.3):
    
    # noise_level float -> 'clean' or 'noise40%' 등
    if noise_level == 0.0:
        noise_str = 'clean'
    else:
        noise_str = f"noise{int(noise_level*100)}%"
    
    # 경로 설정
    config = {
        'example': 'single_pendulum',
        'damping_coef': damping_coeff,
        'noise_level': noise_str
    }
    paths = get_paths(config, folders=['data'])  # fig 폴더는 따로 안 씀, data 폴더에 모두 저장
    
    # 펜듈럼 객체 생성 및 데이터 생성
    pen = SinglePendulum(m=1.0, L=1.0, g=9.81, c=damping_coeff, noise_level=noise_level)
    pen.generate_data(theta_0=0.5, omega_0=0.5, stoptime=20.0, numpoints=20010)
    pen.split_data(train_size=1000, valid_size=200, test_size=4000, timestep=5)
    
    # 그림과 데이터 모두 data 폴더에 저장
    pen.plot_data(save_dir=paths['data'])
    pen.save_data(save_dir=paths['data'])
    
if __name__ == "__main__":

    for noise in [0.0, 0.1, 0.2, 0.3, 0.4]:
        single_pen_damped(noise_level=noise, damping_coeff=0.0)
