import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import cond

from trajax.optimizers import ILQRHyperparams, ILQR

x_dim = 2
u_dim = 1
dt = 0.1
T = 10

g = 9.81
l = 5.0

initial_state = jnp.array([jnp.pi / 2, 0.0])
initial_actions = jnp.zeros(shape=(int(T / dt), u_dim))
control_low = jnp.array([-5.0]).reshape(1, )
control_high = jnp.array([5.0]).reshape(1, )
num_steps = initial_actions.shape[0]


def cost_fn(x, u, t, length):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)

    def running_cost(x, u, t):
        return dt * (jnp.sum(x ** 2) + jnp.sum(u ** 2))

    def terminal_cost(x, u, t):
        return jnp.sum(x ** 2)

    return cond(t == num_steps, terminal_cost, running_cost, x, u, t)


def dynamics_fn(x, u, t, length):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)
    x0 = x[1]
    x1 = u[0] + g / length * jnp.sin(x[0])
    return x + jnp.array([x0, x1]) * dt


ilqr = ILQR(cost_fn, dynamics_fn)

ts = jnp.arange(0, T, dt)

ilqr_params = ILQRHyperparams(maxiter=100)

dynamics_params = jnp.array(5.0)
cost_params = dynamics_params

start_time = time.time()
out = ilqr.solve(dynamics_params, cost_params, initial_state, initial_actions, ilqr_params)
print('Cost: ', out[2])
print("Time taken: ", time.time() - start_time)

start_time = time.time()
out = ilqr.solve(dynamics_params, cost_params, initial_state, initial_actions, ilqr_params)
print('Cost: ', out[2])
print("Time taken: ", time.time() - start_time)

plt.plot(ts, out.xs[:-1, :], label="xs")
plt.title("iLQR warmup")
plt.plot(ts, out.us, label="us")
plt.legend()
plt.show()

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np


def multicolored_lines(fig, ax, x, y, z):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    lc = colorline(x, y, z=z, cmap='Greens')
    fig.colorbar(lc, ax=ax, label=r'$\sum_{i}\sigma_i^2(x(t), u(t))$')
    return fig, ax


def colorline(x, y, z=None, cmap='Greens', linewidth=2, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth, alpha=alpha, label='Trajectory')

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


# create a new figure
fig = plt.figure()

# add a subplot to the figure
ax = fig.add_subplot(1, 1, 1)
ax.grid()

number_of_equidistant_measurements = 4
repeat_num = int(out.xs.shape[0] / number_of_equidistant_measurements)

# plot the trajectory using blue color and line style '-'
fig, ax = multicolored_lines(fig, ax, out.xs[:, 0], out.xs[:, 1], out.xs[:, 0] ** 2 + out.xs[:, 0] ** 2)

ax.scatter(out.xs[:, 0][::repeat_num], out.xs[:, 1][::repeat_num], color='blue', marker='+', s=100, linewidth=2,
           label='Equidistant measurements')

indices = [0, 1, 2, 3, 5, 10, 20, 50, 100]
ax.scatter(out.xs[indices, 0], out.xs[indices, 1], color='red',
           marker='x', s=100, linewidth=2, label='Proposed measurements')

dx_0 = jnp.diff(out.xs[:, 0])
dx_1 = jnp.diff(out.xs[:, 1])

ax.quiver(out.xs[:-1, 0][::5], out.xs[:-1, 1][::5], dx_0[::5], dx_1[::5], angles='xy', scale=1.0, color='black',
          alpha=0.2, linewidth=0)

# set the x and y axis labels
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')

# set the title of the plot
ax.set_title('2D Trajectory')
ax.legend()

# show the plot
plt.show()
