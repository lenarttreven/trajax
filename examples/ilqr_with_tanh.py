import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import cond

from trajax.optimizers import ILQRHyperparams, ILQR

x_dim = 2
u_dim = 1
dt = 0.01
T = 10

g = 9.81
l = 5.0

initial_state = jnp.array([jnp.pi / 2, 0.0])
initial_actions = jnp.zeros(shape=(int(T / dt), u_dim))

max_abs_bound = jnp.array(5.0)

num_steps = initial_actions.shape[0]


def cost_fn(x, u, t, length):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)

    def running_cost(x, u, t):
        return dt * (jnp.sum(x ** 2) + jnp.sum((jnp.tanh(u) * max_abs_bound) ** 2))

    def terminal_cost(x, u, t):
        return jnp.sum(x ** 2)

    return cond(t == num_steps, terminal_cost, running_cost, x, u, t)


def dynamics_fn(x, u, t, length):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)
    x0 = x[1]
    x1 = jnp.tanh(u[0]) * max_abs_bound + g / length * jnp.sin(x[0])
    return x + jnp.array([x0, x1]) * dt


ilqr = ILQR(cost_fn, dynamics_fn)

ts = jnp.arange(0, T, dt)

ilqr_params = ILQRHyperparams(maxiter=100,  make_psd=True)

dynamics_params = jnp.array(5.0)
cost_params = dynamics_params

start_time = time.time()
out = ilqr.solve(dynamics_params, cost_params, initial_state, initial_actions, ilqr_params)
print('Cost: ', out.obj)
print("Time taken: ", time.time() - start_time)

start_time = time.time()
out = ilqr.solve(dynamics_params, cost_params, initial_state, initial_actions, ilqr_params)
print('Cost: ', out.obj)
print("Time taken: ", time.time() - start_time)

plt.plot(ts, out.xs[:-1, :], label="xs")
plt.title("iLQR warmup")
plt.plot(ts, jnp.tanh(out.us), label="us")
plt.legend()
plt.show()
