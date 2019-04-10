from __future__ import division
from __future__ import print_function, division
import jax.numpy as jaxnp
import numpy as np
from jax import grad, jit
import numpy.random as rnd
from jax import jacfwd, jacrev
import scipy
import matplotlib.pyplot as plt

np.random.seed(1)


def hessian(fun):
    return jit(jacfwd(jacrev(fun)))


def min_R(R, d, x0):
    """Returns argmin_x(R(x))"""
    R_grad = jit(grad(R))
    R_hess = jit(hessian(R))
    steps = 100
    tol = 1e-5
    x = x0
    for i in range(steps):
        # Solve x_{n+1} = x_n - hess(x)^{-1}*grad(x)
        diff = np.linalg.solve(R_hess(x), -R_grad(x))
        x = x + diff
        if np.linalg.norm(diff) < tol:
            break
    return x


def sample_spherical(d):
    """Generate a sample uniformly at random from the surface of the 
    unit hypersphere. We generate a vector with each component a random
    normal and then normalize."""
    vector = rnd.normal(loc=0, scale=1, size=d)
    return vector / np.linalg.norm(vector, axis=0)


def inner_opt(g_tau, x_tau, R, sigma, eta, warm_start):
    """Returns the objective to be minimized in the inner loop of the
    algorithm. Takes:
    g_tau: Nxd array where each row is g_t for t=1:tau
    x_tau: Nxd array where each row is x_t for t=1:tau
    R: function R^d: -> R
    Returns argmin_x [g_tau*x + 1/2 * sigma * ||x-x_tau||^2 + (1/eta)R(x)]
    """
    x = warm_start
    steps = 100
    tol = 1e-5
    objective = lambda x: (jaxnp.sum(jaxnp.dot(g_tau, x)) +
                           0.5*sigma * jaxnp.sum((x - x_tau)**2)
                           + (1/eta) * R(x))
    objective_grad = jit(grad(objective))
    objective_hess = jit(hessian(objective))
    for i in range(steps):
        # Solve x_{n+1} = x_n - hess(x)^{-1}*grad(x)
        diff = np.linalg.solve(objective_hess(x), -objective_grad(x))
        x = x + diff
        if np.linalg.norm(diff) < tol:
            break
    return x


def optimize(f, T, R, d, lr_params=None):
    """Runs the Hazan-Levy method for T timesteps.
    Input:
    f:    R^d -> R    Objective function
    T:    int > 0     Sample budget
    R:    R^d -> R    Nu-self-concordant barrier function
    d:    int > 0     Problem dimension
    lr_params dict    Dictionary with params to determine eta, beta, sigma
      nu    R>0    self-concordancy parameter
      beta  R>0    smoothness of the objective
      sigma R>0    strong-convexity of the objective
      L     R>0    maximum of the objective

    At each iteration we need to minimize over a certain function.
    We pass this off to the function inner_opt, which uses Newton's method
    to solve this problem.
    """

    if lr_params is None:
        eta0 = np.sqrt(3/(50*d**3))
        sigma = 1
    else:
        sigma = lr_params['sigma']
        eta0 = np.sqrt((lr_params['nu']
                        + 2*lr_params['beta']/lr_params['sigma'])
                       / (2*d**2 * lr_params['L']**2))

    eta0 = eta0 / 100
    # Find x_0 = argmin_x(R(x))
    x_t = min_R(R, d, np.array([0.]*d))
    print('Initial x is {}'.format(x_t))
    xs = np.ndarray((T, d))
    xs[0] = x_t
    g_tau = np.ndarray((T, d))
    x_tau = np.ndarray((T+1, d))
    x_tau[0, :] = x_t

    hess_R = hessian(R)
    for t in range(1, T+1):
        eta = eta0 * (np.log(t+1) / np.sqrt(t+1))
        B_t = np.linalg.inv(scipy.linalg.sqrtm(hess_R(x_t)
                                               + eta * sigma * t * np.eye(d)))
        B_t_inv = scipy.linalg.sqrtm(hess_R(x_t) + eta * sigma * t * np.eye(d))
        u = sample_spherical(d)
        val = f(x_t + np.dot(B_t, u))
        g_t = d * np.dot(np.dot(val, B_t_inv), u)
        g_tau[t-1, :] = g_t
        x_t = inner_opt(g_tau[:t, :], x_tau[:t, :], R, sigma, eta,
                        warm_start=x_t)
        print("Sampling at point {} at timestep {}\n----".format(x_t, t))
        x_tau[t, :] = x_t
    return x_t, x_tau


def test_quadratic(r0, d):
    """Test the optimization procedure with a noisy quadratic
    objective, f(x) = ||x||^2 + N(0, 0.01)"""
    f = lambda x: np.linalg.norm(x) ** 2 + rnd.normal(loc=0, scale=0.01)
    R = lambda x: -jaxnp.log(r0**2 - jaxnp.sum(x**2))
    lr_params = {'nu': 1, 'beta': 1, 'sigma': 1, 'L': r0**2}
    # R = lambda x: full_R(3, x)
    return optimize(f, 100, R, d, lr_params)


def test_quadratic_constrained(r0, d):
    """Test the optimization procedure with a noisy quadratic
    objective, f(x) = ||x||^2 + N(0, 0.01),
    and a constraint so that the optimum is on the boundary"""
    f = lambda x: np.linalg.norm(x) ** 2 + rnd.normal(loc=0, scale=0.01)
    R = lambda x: -jaxnp.log(r0**2 - jaxnp.sum((x - np.ones(d)*(1.2)*r0)**2))
    lr_params = {'nu': 1, 'beta': 1, 'sigma': 1, 'L': r0**2}
    # R = lambda x: full_R(3, x)
    return optimize(f, 200, R, d, lr_params)


outs = test_quadratic(10, 3)
xs = outs[-1]
norms = [np.linalg.norm(x) for x in xs]
plt.plot(range(len(norms)), norms)
plt.show()

# outs = test_quadratic_constrained(3, 10)
# xs = outs[-1]
# norms = [np.linalg.norm(x) for x in xs]
# plt.plot(range(len(norms)), norms)
# plt.show()
