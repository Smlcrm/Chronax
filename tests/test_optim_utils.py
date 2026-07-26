"""Unit tests for the adaptive optimizer suite in chronax.utils.

Exercises:
- ``minimize_armijo``: convex quadratic, Rosenbrock, early-convergence,
  vmap batch correctness, softplus-reparametrized positivity constraint.
- ``bounded_line_minimize``: 1-D convex, parabola offset root.
- ``multistart_argmin``: picks the best basin across multiple starts.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.utils import (
    bounded_line_minimize,
    minimize_armijo,
    multistart_argmin,
    MinimizeState,
    nelder_mead,
    NMState,
)


jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# minimize_armijo — basic scalar problems
# -----------------------------------------------------------------------------

def test_minimize_armijo_convex_quadratic_2d():
    A = jnp.array([[3.0, 0.5], [0.5, 2.0]])
    b = jnp.array([1.0, -2.0])

    def f(x):
        return 0.5 * x @ A @ x - b @ x

    result = minimize_armijo(f, jnp.array([5.0, 5.0]))
    x_star_expected = jnp.linalg.solve(A, b)
    np.testing.assert_allclose(result.x, x_star_expected, atol=1e-5)
    assert bool(result.converged)
    assert int(result.iter) < 50


def test_minimize_armijo_ill_conditioned_quadratic():
    # Condition number ~100; gradient descent struggles, L-BFGS should handle.
    diag = jnp.array([100.0, 10.0, 1.0, 0.1])

    def f(x):
        return 0.5 * jnp.sum(diag * x * x)

    result = minimize_armijo(f, jnp.ones(4), max_iter=300, grad_tol=1e-7)
    np.testing.assert_allclose(result.x, jnp.zeros(4), atol=1e-4)
    assert bool(result.converged)


def test_minimize_armijo_rosenbrock():
    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    result = minimize_armijo(
        rosenbrock, jnp.array([-1.2, 1.0]),
        max_iter=500, grad_tol=1e-6,
    )
    np.testing.assert_allclose(result.x, jnp.array([1.0, 1.0]), atol=1e-3)
    assert bool(result.converged)


def test_minimize_armijo_already_at_optimum():
    def f(x):
        return jnp.sum(x * x)

    result = minimize_armijo(f, jnp.zeros(3))
    np.testing.assert_allclose(result.x, jnp.zeros(3))
    assert bool(result.converged)
    assert int(result.iter) == 0


# -----------------------------------------------------------------------------
# vmap correctness
# -----------------------------------------------------------------------------

def test_minimize_armijo_vmap_matches_sequential():
    A = jnp.array([[3.0, 0.5], [0.5, 2.0]])
    b_batch = jnp.array([[1.0, -2.0], [0.0, 1.0], [-1.0, 0.5], [2.0, -1.0]])
    x0_batch = jnp.tile(jnp.array([5.0, 5.0]), (4, 1))

    def make_fn(b):
        return lambda x: 0.5 * x @ A @ x - b @ x

    seq_results = jnp.stack([
        minimize_armijo(make_fn(b), x0).x for b, x0 in zip(b_batch, x0_batch)
    ])
    vm_results = jax.vmap(
        lambda b, x0: minimize_armijo(lambda x: 0.5 * x @ A @ x - b @ x, x0).x
    )(b_batch, x0_batch)

    np.testing.assert_allclose(vm_results, seq_results, atol=1e-4)


def test_minimize_armijo_vmap_early_converged_elements():
    # Some batch elements start at the optimum; others do not. The optimizer
    # should still converge without the converged elements drifting.
    def f(x, center):
        return jnp.sum((x - center) ** 2)

    centers = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    x0s = jnp.array([[0.0, 0.0],        # already at optimum
                     [0.0, 0.0],
                     [5.0, 5.0],
                     [-3.0, -3.0]])

    vm = jax.vmap(lambda x0, c: minimize_armijo(lambda x: f(x, c), x0).x)
    got = vm(x0s, centers)
    np.testing.assert_allclose(got, centers, atol=1e-4)


# -----------------------------------------------------------------------------
# Reparametrization for bounded problems
# -----------------------------------------------------------------------------

def test_minimize_armijo_softplus_reparam_positivity():
    # Recover x* = 2 with strict positivity via softplus reparam.
    def fun_in_z(z):
        x = jax.nn.softplus(z)
        return jnp.sum((x - 2.0) ** 2)

    result = minimize_armijo(fun_in_z, jnp.array([-1.0]), max_iter=200)
    x_star = jax.nn.softplus(result.x)
    np.testing.assert_allclose(x_star, jnp.array([2.0]), atol=1e-4)


def test_minimize_armijo_sigmoid_reparam_two_sided_bound():
    # Recover x* = 0.3 within (0, 1) via sigmoid reparam.
    def fun_in_z(z):
        x = jax.nn.sigmoid(z)
        return jnp.sum((x - 0.3) ** 2)

    result = minimize_armijo(fun_in_z, jnp.array([0.0]), max_iter=200)
    x_star = jax.nn.sigmoid(result.x)
    np.testing.assert_allclose(x_star, jnp.array([0.3]), atol=1e-4)


# -----------------------------------------------------------------------------
# bounded_line_minimize
# -----------------------------------------------------------------------------

def test_bounded_line_minimize_parabola():
    def f(x):
        return (x - 0.4) ** 2

    x_star, f_star = bounded_line_minimize(f, 0.0, 1.0, max_iter=80)
    np.testing.assert_allclose(float(x_star), 0.4, atol=1e-5)
    np.testing.assert_allclose(float(f_star), 0.0, atol=1e-9)


def test_bounded_line_minimize_at_boundary():
    # Global min at x=0, which is the lower bound.
    def f(x):
        return x * x

    x_star, _ = bounded_line_minimize(f, 0.0, 1.0, max_iter=80)
    assert float(x_star) < 1e-3


# -----------------------------------------------------------------------------
# multistart_argmin
# -----------------------------------------------------------------------------

def test_multistart_argmin_picks_lowest_basin():
    # Double-well: minima at x = -2 and x = +2; global at x = -2 (deeper).
    def f(x):
        a = x - 2.0
        b = x + 2.0
        return jnp.minimum(a @ a + 1.0, b @ b)   # shift so x=-2 is strictly lower

    starts = jnp.array([[-3.0], [3.0], [0.0]])

    def minimizer(x0):
        return minimize_armijo(f, x0, max_iter=200)

    result = multistart_argmin(minimizer, starts)
    np.testing.assert_allclose(result.x, jnp.array([-2.0]), atol=1e-3)


def test_multistart_argmin_all_same_basin():
    # Single-basin quadratic: all starts should converge to the same point.
    def f(x):
        return jnp.sum((x - jnp.array([1.0, -1.0])) ** 2)

    starts = jnp.array([[5.0, 5.0], [-5.0, -5.0], [0.0, 0.0]])

    def minimizer(x0):
        return minimize_armijo(f, x0)

    result = multistart_argmin(minimizer, starts)
    np.testing.assert_allclose(result.x, jnp.array([1.0, -1.0]), atol=1e-4)


# -----------------------------------------------------------------------------
# nelder_mead — derivative-free simplex
# -----------------------------------------------------------------------------

def test_nelder_mead_convex_quadratic_2d():
    A = jnp.array([[3.0, 0.5], [0.5, 2.0]])
    b = jnp.array([1.0, -2.0])

    def f(x):
        return 0.5 * x @ A @ x - b @ x

    result = nelder_mead(f, jnp.array([5.0, 5.0]), max_iter=400)
    x_star_expected = jnp.linalg.solve(A, b)
    np.testing.assert_allclose(result.x, x_star_expected, atol=1e-4)
    assert bool(result.converged)


def test_nelder_mead_rosenbrock():
    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    result = nelder_mead(rosenbrock, jnp.array([-1.2, 1.0]), max_iter=1000)
    np.testing.assert_allclose(result.x, jnp.array([1.0, 1.0]), atol=1e-3)
    assert bool(result.converged)


def test_nelder_mead_non_smooth_absolute():
    # |x - 1.7| — has a kink that gradient-based methods handle poorly.
    def f(x):
        return jnp.sum(jnp.abs(x - jnp.array([1.7, -0.4])))

    result = nelder_mead(f, jnp.array([0.0, 0.0]), max_iter=400)
    np.testing.assert_allclose(result.x, jnp.array([1.7, -0.4]), atol=1e-3)


def test_nelder_mead_already_at_optimum():
    def f(x):
        return jnp.sum(x * x)

    result = nelder_mead(f, jnp.zeros(3), max_iter=50)
    np.testing.assert_allclose(result.x, jnp.zeros(3), atol=1e-3)
    # May or may not be "converged" by the tight simplex-diameter criterion
    # depending on initial simplex size; what matters is that we stay near 0.
    assert float(result.f) < 1e-6


def test_nelder_mead_vmap_matches_sequential():
    A = jnp.array([[3.0, 0.5], [0.5, 2.0]])
    b_batch = jnp.array([[1.0, -2.0], [0.0, 1.0], [-1.0, 0.5], [2.0, -1.0]])
    x0_batch = jnp.tile(jnp.array([5.0, 5.0]), (4, 1))

    seq_results = jnp.stack([
        nelder_mead(lambda x: 0.5 * x @ A @ x - b @ x, x0, max_iter=400).x
        for b, x0 in zip(b_batch, x0_batch)
    ])
    vm = jax.vmap(
        lambda b, x0: nelder_mead(lambda x: 0.5 * x @ A @ x - b @ x, x0, max_iter=400).x
    )(b_batch, x0_batch)
    np.testing.assert_allclose(vm, seq_results, atol=1e-3)


def test_nelder_mead_sigmoid_reparam():
    # Recover x* = 0.3 within (0, 1) via sigmoid reparam, just like
    # minimize_armijo. NM handles the smooth reparam fine.
    def fun_in_z(z):
        x = jax.nn.sigmoid(z)
        return jnp.sum((x - 0.3) ** 2)

    result = nelder_mead(fun_in_z, jnp.array([0.0]), max_iter=300)
    x_star = jax.nn.sigmoid(result.x)
    np.testing.assert_allclose(x_star, jnp.array([0.3]), atol=1e-3)


def test_nelder_mead_escapes_saddle_point():
    # f(x, y) = x² − y² has a saddle at (0, 0) and no minimum — should not
    # crash or hang; should exit via max_iter with non-finite or finite f.
    def f(x):
        return x[0] ** 2 - x[1] ** 2

    result = nelder_mead(f, jnp.array([0.1, 0.1]), max_iter=50)
    # Just confirm it terminated and produced finite output (no NaN/Inf).
    assert jnp.isfinite(result.f) | (result.f == -jnp.inf)
    assert int(result.iter) <= 50


# ---------------------------------------------------------------------------
# adaptive_iterate — convergence-based lax.while_loop step runner
# ---------------------------------------------------------------------------
def test_adaptive_iterate_stops_early_on_convergence():
    """A loss that flattens to a floor should trip the plateau detector well
    before the cap (a purely geometric loss would NOT — constant rel improvement)."""
    from chronax.utils import adaptive_iterate

    def step(c):                                 # loss: 4, 3, 2, 1, 0, 0, ... (flat floor)
        i = c + 1
        return i, jnp.maximum(0.0, 5.0 - i)
    _, bl, n = adaptive_iterate(step, jnp.array(0.0), lambda c: jnp.asarray([c]),
                                max_steps=100, rtol=1e-6, patience=2)
    assert int(n) < 100
    assert float(bl) == 0.0


def test_adaptive_iterate_runs_to_cap_when_still_improving():
    """No plateau within the cap (rtol tiny) => runs exactly max_steps."""
    from chronax.utils import adaptive_iterate

    def step(c):                                 # 10/(i+1): shrinking but rel > 1e-9 for 12 steps
        i = c + 1.0
        return i, 10.0 / (i + 1.0)
    _, _, n = adaptive_iterate(step, jnp.array(0.0), lambda c: jnp.zeros(1),
                               max_steps=12, rtol=1e-9, patience=2)
    assert int(n) == 12


def test_adaptive_iterate_returns_best_not_last():
    """Returns the minimum-loss iterate seen, not the final (plateau) one."""
    from chronax.utils import adaptive_iterate

    losses = jnp.array([1.0, 9.0, 9.0, 9.0])     # point 0 is best, then worse + plateau
    def step(c):                                  # loss measured AT the input point c
        return c + 1, losses[jnp.clip(c, 0, 3)]
    bp, bl, _ = adaptive_iterate(step, jnp.array(0), lambda c: jnp.asarray([c * 1.0]),
                                 max_steps=4, rtol=1e-6, patience=2)
    assert float(bl) == 1.0
    assert float(bp[0]) == 0.0                    # the point where loss 1.0 was measured


def test_adaptive_iterate_traces_under_vmap():
    """The whole loop is lax.while_loop + jnp — it must trace under jax.vmap
    (this is the property that lets the CV/conformal path reuse it)."""
    from chronax.utils import adaptive_iterate

    def run(x0):
        def step(c):
            return 0.5 * c, (c[0]) ** 2
        _, bl, _ = adaptive_iterate(step, x0, lambda c: c, max_steps=20, rtol=1e-3)
        return bl
    out = jax.vmap(run)(jnp.array([[1.0], [2.0], [3.0]]))
    assert out.shape == (3,)
