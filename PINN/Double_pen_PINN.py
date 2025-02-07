import deepxde as dde
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Masses
m1 = 2.0
m2 = 0.5
# gravity
g = 9.81
# Natural lengths
L1 = 1
L2 = 2

# Initial conditions
# d1_0 and d2_0 are the initial displacements; v1_0 and v2_0 are the initial velocities
d1_0 = 1
d2_0 = 0.5

v1_0 = 0.0
v2_0 = 0.3

b1 = 0
b2 = 0

# maximum time to simulate
t_max = 4

def dy(t, x):
    return dde.grad.jacobian(x, t)

def pde(t, x):
    # mass 1 location
    x_1 = x[:, 0:1]
    # mass 2 location
    x_2 = x[:, 1:2]

    dx1_t = dde.grad.jacobian(x, t, i = 0, j = 0)
    dx2_t = dde.grad.jacobian(x, t, i = 1, j = 0)

    dx1_tt = dde.grad.hessian(x, t, i = 0, j = 0, component = 0)
    dx2_tt = dde.grad.hessian(x, t, i = 0, j = 0, component = 1)

    pde1 = (m1 + m2) * L1 * dx1_tt + m2 * L2 * dx2_tt * tf.cos(x_1-x_2) + m2 * L2 * dx2_t ** 2 * tf.sin(x_1-x_2) + (m1 + m2) * g * tf.sin(x_1)
    pde2 = m2 * L2 * dx2_tt + m2 * L1 * dx1_tt * tf.cos(x_1-x_2) - m2 * L1 * dx1_t ** 2 * tf.sin(x_1-x_2) + m2 * g * tf.sin(x_2)

    return [pde1, pde2]

def boundary_init(t, on_boundary):
    return on_boundary and np.isclose(t[0], 0)

geom = dde.geometry.Interval(0, t_max)

init_d1 = dde.icbc.PointSetBC(np.array([0]), np.array([d1_0]).reshape(-1, 1), component=0)
init_d2 = dde.icbc.PointSetBC(np.array([0]), np.array([d2_0]).reshape(-1, 1), component=1)
init_v1 = dde.OperatorBC(geom, lambda x, y, _: dy(x, y[:, 0:1]), boundary_init)
init_v2 = dde.OperatorBC(geom, lambda x, y, _: dy(x, y[:, 1:2])-0.3, boundary_init)


data = dde.data.PDE(geom,
                    pde,
                    [init_d1, init_d2, init_v1, init_v2],
                    num_domain = 2000,
                    num_boundary = 200,
                    num_test = 2000)

def sine_activation(x):
    return tf.math.sin(x)

layer_size = [1] + [64] * 10 + [2]
activation = "tanh"
initializer = "Glorot uniform"

net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr = 1e-3)

losshistory, train_state = model.train(epochs = 12000)
dde.saveplot(losshistory, train_state, issave = False, isplot = False)

dde.optimizers.config.set_LBFGS_options(maxiter=15000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave = True, isplot = True)

# Parameters
# Masses
m1 = 2.0
m2 = 0.5
# gravity
g = 9.81
# Natural lengths
L1 = 1
L2 = 2

# Initial conditions
# d1_0 and d2_0 are the initial displacements; v1_0 and v2_0 are the initial velocities
d1_0 = 1
d2_0 = 0.5

v1_0 = 0.0
v2_0 = 0.3

# Time span
t_span = np.linspace(0, 10, 2000)

# Initial conditions vector
initial_conditions = [d1_0, v1_0, d2_0, v2_0]

# ODE function
def double_pendulum_ode(y, t, m1, m2, L1, L2, g):
    dydt = np.zeros(4)

    # Extracting variables
    d1, v1, d2, v2 = y

    # Equations of motion
    dydt[0] = v1
    dydt[1] = (-g * (2 * m1 + m2) * np.sin(d1) - m2 * g * np.sin(d1 - 2 * d2) - 2 * np.sin(d1 - d2) * m2 * (v2**2 * L2 + v1**2 * L1 * np.cos(d1 - d2))) / (L1 * (2 * m1 + m2 - m2 * np.cos(2 * d1 - 2 * d2)))
    dydt[2] = v2
    dydt[3] = (2 * np.sin(d1 - d2) * (v1**2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(d1) + v2**2 * L2 * m2 * np.cos(d1 - d2))) / (L2 * (2 * m1 + m2 - m2 * np.cos(2 * d1 - 2 * d2)))

    return dydt
# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = t_max
numpoints = 250
t = geom.random_points(5000)
t_test = np.linspace(0,10,2000)
t_reshaped = t_test.reshape(-1, 1)

t[:,0].sort()
# Solve ODE using odeint
sol = odeint(double_pendulum_ode, initial_conditions, t_span, args=(m1, m2, L1, L2, g))

# Extracting angles
theta1 = sol[:, 0]
theta2 = sol[:, 2]

# Extracting angles
theta1 = sol[:, 0]
theta2 = sol[:, 2]
print(t_span.shape)

# Numerical differentiation to compute velocities (first derivative)
theta1_vel = np.gradient(theta1, t_span)  # First derivative of theta1
theta2_vel = np.gradient(theta2, t_span)  # First derivative of theta2

# Numerical differentiation to compute accelerations (second derivative)
theta1_accel = np.gradient(theta1_vel, t_span)  # Second derivative of theta1 (acceleration)
theta2_accel = np.gradient(theta2_vel, t_span)  # Second derivative of theta2 (acceleration)

# Plotting the accelerations
plt.figure(figsize=(12, 6))

# Acceleration of theta1
plt.plot(t_span, theta1_accel, label=r'$\ddot{\Theta}_1$ (Acceleration)', linestyle='dashed', color='r', lw=2)

# Acceleration of theta2
plt.plot(t_span, theta2_accel, label=r'$\ddot{\Theta}_2$ (Acceleration)', linestyle='dashed', color='b', lw=2)

# Labels and plot settings
plt.legend(loc='best', fontsize=15)
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Acceleration (rad/s²)', fontsize=15)
plt.grid(True)
plt.show()

result = model.predict(t_reshaped)
output_path = "Doublepen_predictions.npy"

t_grad = t_reshaped.reshape(-1)

# NumPy 배열로 저장
np.save(output_path, result)
usol1 = np.array(result[:, 0])
usol2 = np.array(result[:, 1])

print(t_grad.shape)


# Numerical differentiation to compute velocities (first derivative)
usol1_vel = np.gradient(usol1, t_grad)  # First derivative of theta1
usol2_vel = np.gradient(usol2, t_grad)  # First derivative of theta2

# Numerical differentiation to compute accelerations (second derivative)
usol1_accel = np.gradient(usol1_vel, t_grad)  # Second derivative of theta1 (acceleration)
usol2_accel = np.gradient(usol2_vel, t_grad)  # Second derivative of theta2 (acceleration)

# Plotting the accelerations
plt.figure(figsize=(12, 6))

# Acceleration of theta1
plt.plot(t_span, usol1_accel, label=r'$\ddot{\Theta}_1$ (Acceleration)', linestyle='dashed', color='r', lw=2)

# Acceleration of theta2
plt.plot(t_span, usol2_accel, label=r'$\ddot{\Theta}_2$ (Acceleration)', linestyle='dashed', color='b', lw=2)

# Labels and plot settings
plt.legend(loc='best', fontsize=15)
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Acceleration (rad/s²)', fontsize=15)
plt.grid(True)
plt.show()
lw = 2

# 중간 지점을 자동으로 계산
half = len(t_span) // 2

# Plotting the accelerations pred
plt.figure(figsize=(12, 6))

# Acceleration of theta1 (interpolation)
plt.plot(t_span[:half], usol1_accel[:half], label=r'$\ddot{\Theta}_1$ (PINN)', linestyle='dashed', color='r', lw=2)

# Acceleration of theta1 (extrapolation)
plt.plot(t_span[half:], usol1_accel[half:], label=r'$\ddot{\Theta}_1$ (PINN_ext)', linestyle='dashed', color='g', lw=2)

# Acceleration of theta2 (interpolation)
plt.plot(t_span[:half], usol2_accel[:half], label=r'$\ddot{\Theta}_2$ (PINN)', linestyle='dashed', color='b', lw=2)

# Acceleration of theta2 (extrapolation)
plt.plot(t_span[half:], usol2_accel[half:], label=r'$\ddot{\Theta}_2$ (PINN_ext)', linestyle='dashed', color='g', lw=2)


# Plotting the accelerations Reference

# Acceleration of theta1
plt.plot(t_span, theta1_accel, label=r'$\ddot{\Theta}_1$ (Ref)',  color='k', lw=2)

# Acceleration of theta2
plt.plot(t_span, theta2_accel, label=r'$\ddot{\Theta}_2$ (Ref)',  color='k', lw=2)

# Labels and plot settings
plt.legend(loc='best', fontsize=15)
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Acceleration (rad/s²)', fontsize=15)
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.title('PINN prediction',fontsize='20')

# Plot ground truth data for Mass1
plt.plot(t_reshaped[:, 0], theta1, alpha=1, label=r'$\theta_1$(ODE solver)', c='k', lw=lw)

# Plot ground truth data for Mass2
plt.plot(t_reshaped[:, 0], theta2, alpha=1, label=r'$\theta_2$(ODE solver)', c='k', lw=lw)

# 중간 지점을 자동으로 계산
half = len(t_reshaped) // 2

# Plot predicted data for Mass1 (interpolation)
plt.plot(t_reshaped[:, 0][:half], usol1[:half], alpha=1, label=r'$\theta_1$(PINN interpol)', c='r', lw=lw, linestyle='dashed')

# Plot predicted data for Mass1 (extrapolation)
plt.plot(t_reshaped[:, 0][half:], usol1[half:], alpha=1, label=r'$\theta_1$(PINN extrapol)', c='g', lw=lw, linestyle='dashed')

# Plot predicted data for Mass2 (interpolation)
plt.plot(t_reshaped[:, 0][:half], usol2[:half], alpha=1, label=r'$\theta_2$(PINN interpol)', c='b', lw=lw, linestyle='dashed')

# Plot predicted data for Mass2 (extrapolation)
plt.plot(t_reshaped[:, 0][half:], usol2[half:], alpha=1, label=r'$\theta_2$(PINN extrapol)', c='g', lw=lw, linestyle='dashed')


plt.legend(fontsize=15, loc='best')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Angle (rad)', fontsize=20)
plt.grid(True)
plt.show()


