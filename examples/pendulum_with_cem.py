import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import cond

from trajax.optimizers import CEMHyperparams, cem

x_dim = 2
u_dim = 1
dt = 0.01
T = 10

g = 9.81
l = 5.0

initial_state = jnp.array([jnp.pi / 2, 0.0])
initial_actions = jnp.zeros(shape=(int(T / dt), u_dim))
control_low = jnp.array([-5.0]).reshape(1, )
control_high = jnp.array([5.0]).reshape(1, )
num_steps = initial_actions.shape[0]


def cost_fn(x, u, t):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)

    def running_cost(x, u, t):
        return dt * (jnp.sum(x ** 2) + jnp.sum(u ** 2))

    def terminal_cost(x, u, t):
        return jnp.sum(x ** 2)

    return cond(t == num_steps, terminal_cost, running_cost, x, u, t)


def dynamics_fn(x, u, t):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)
    x0 = x[1]
    x1 = u[0] + g / l * jnp.sin(x[0])
    return x + jnp.array([x0, x1]) * dt


ts = jnp.arange(0, T, dt)

start_time = time.time()
cem_params = CEMHyperparams(max_iter=10, sampling_smoothing=0.0, num_samples=500, evolution_smoothing=0.0,
                            elite_portion=0.1)

out = cem(cost_fn, dynamics_fn, initial_state, initial_actions, control_low=control_low, control_high=control_high,
          hyperparams=cem_params)
print('Cost: ', out[2])
print("Time taken: ", time.time() - start_time)

start_time = time.time()
cem_params = CEMHyperparams(max_iter=10, sampling_smoothing=0.0, num_samples=500, evolution_smoothing=0.0,
                            elite_portion=0.1)

out = cem(cost_fn, dynamics_fn, initial_state, initial_actions, control_low=control_low, control_high=control_high,
          hyperparams=cem_params)
print('Cost: ', out[2])
print("Time taken: ", time.time() - start_time)

plt.plot(ts, out[0][:-1, :], label="xs")
plt.title("iLQR warmup")
plt.plot(ts, out[1], label="us")
plt.legend()
plt.show()
