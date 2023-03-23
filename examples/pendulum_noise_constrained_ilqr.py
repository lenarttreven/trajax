import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import cond

from trajax.optimizers import constrained_ilqr

x_dim = 2
u_dim = 1
dt = 0.1
T = 10

g = 9.81
l = 5.0

initial_state = jnp.array([jnp.pi / 2, 0.0])
initial_actions = jnp.zeros(shape=(int(T / dt), u_dim + x_dim))

num_steps = initial_actions.shape[0]


def cost_fn(x, u, t):
    assert x.shape == (x_dim,) and u.shape == (u_dim + x_dim,)

    def running_cost(x, u, t):
        return dt * (jnp.sum(x ** 2) + jnp.sum(u[:u_dim] ** 2))

    def terminal_cost(x, u, t):
        return jnp.sum(x ** 2)

    return cond(t == num_steps, terminal_cost, running_cost, x, u, t)


std = 1e-3


def ode(x, u, eta):
    assert x.shape == (x_dim,) and u.shape == (u_dim,) and eta.shape == (x_dim,)
    x0 = x[1]
    x1 = u[0] + g / l * jnp.sin(x[0])
    x_dot = jnp.array([x0, x1])
    return x_dot + eta * std


x = initial_state
u = jnp.zeros(shape=(u_dim,))
eta = jnp.array([-5.1, 2.3])

x_dot = ode(x, u, eta)


def dynamics_fn(x, u, t):
    assert x.shape == (x_dim,) and u.shape == (u_dim + x_dim,)
    x_dot = ode(x, u[:u_dim], u[u_dim:])
    return x + x_dot * dt


ts = jnp.arange(0, T, dt)

start_time = time.time()

control_low = jnp.array([-1.0, -1.0]).reshape(2, )
control_high = jnp.array([1.0, 1.0]).reshape(2, )


def inequality_constraint_fn(x, u, t):
    assert x.shape == (x_dim,) and u.shape == (u_dim + x_dim,)
    eta = u[u_dim:]
    return jnp.concatenate([eta - control_high, control_low - eta])


out = constrained_ilqr(cost_fn, dynamics_fn, initial_state, initial_actions,
                       inequality_constraint=inequality_constraint_fn)
print("Time taken: ", time.time() - start_time)

xs = out[0]
us = out[1]

plt.plot(ts, xs[:-1, :], label="xs")
plt.title("CEM")
plt.plot(ts, us[:, :u_dim], label="us")
plt.legend()
plt.show()
