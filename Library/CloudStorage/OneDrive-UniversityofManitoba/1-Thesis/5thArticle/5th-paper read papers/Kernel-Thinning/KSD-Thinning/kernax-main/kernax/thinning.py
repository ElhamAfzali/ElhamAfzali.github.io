# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import jax
import jax.numpy as jnp
import numpy as np

from typing import Callable, Optional
from kernax import kernels
from kernax.utils import median_heuristic
from kernax.types import Array


class SteinThinning:
    """Implements the user interface for the vanilla Stein thinning algorithm that relies on the Langevin Stein operator and the IMQ kernel function.

    [1] Riabiz, M., Chen, W. Y., Cockayne, J., Swietach, P., Niederer, S. A., Mackey, L., & Oates, C. J. (2022).
    Optimal thinning of MCMC output. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(4), 1059-1081.

    Examples
    --------

    Assume that we want to thin a two-dimensional Gaussian output.

    .. code::

        import jax

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (10_000,2))

    We need to define the log-target function.

    .. code::

        from jax.scipy.stats import multivariate_normal

        logprob_fn = multivariate_normal.logpdf

    In this simple example, we could have implemented `logprob_fn` ourselves as follows.

    .. code::

        import jax.numpy as jnp

        logprob_fn = lambda x: -0.5*jnp.sum(x**2)

    Next, we compute a lengthscale and the scores.

    .. code::

        from kernax.utils import median_heuristic
        lengthscale = median_heuristic(x)

        score_fn = lambda x: -x
        score_p = jax.vmap(score_fn, 0)(x)

    We can now get the callable function that performs Stein thinning and compresses the Gaussian output.

    .. code::

        stein_thinning_fn = SteinThinning(x, score_p, lengthscale)

        idx = stein_thinning_fn(m = 1_000)

    The selected indices are gathered in the array `idx`.


    Parameters
    ----------

    x
        Sample matrix of size :math:`(n, d)`
    score_p
        Matrix of size :math:`(n, d)` gathering the score values of :math:`x`
    lengthscale
        Scalar lengthscale that will be used for the underlying IMQ kernel


    Returns
    -------

    A callable function which take a single argument :math:`m` and returns the list of the :math:`m` selected indices.

    """

    def __new__(
        cls,
        x: Array,
        score_p: Array,
        lengthscale: Optional[float] = None
    ) -> Callable:

        if lengthscale is None:
            lengthscale = jnp.array(median_heuristic(x))

        stein_fn = jax.tree_util.Partial(kernels.SteinIMQ, lengthscale=lengthscale)
        # stein_fn = kernels.GetSteinFn(jax.tree_util.Partial(kernels.IMQ, lengthscale=lengthscale))
        kpmap = jax.vmap(stein_fn, (None, None, 0, 0))

        def stein_thinning_fn(m: int) -> Array:
            def step_fn(carry, el):
                obj, idx = carry
                kp = kpmap(x[idx], score_p[idx], x, score_p)
                obj += 2.0*kp
                new_idx = jnp.argmin(obj)
                return (obj, new_idx), new_idx
            init_obj = jax.vmap(lambda x1, s1: stein_fn(x1, s1, x1, s1), (0, 0))(x, score_p)
            init_idx = jnp.argmin(init_obj)
            _, idx = jax.lax.scan(step_fn, (init_obj, init_idx), np.arange(1, m))
            return jnp.append(init_idx, idx)

        return stein_thinning_fn


class RegularizedSteinThinning:
    """Implements the user interface for the regularized Stein thinning algorithm using the Langevin Stein operator and the IMQ kernel function.

    [1] Bénard, C., Staber, B., & Da Veiga, S. (2023).
    Kernel Stein Discrepancy thinning: a theoretical perspective of pathologies and a practical fix with regularization. NeurIPS 2023.

    Examples
    --------

    Assume that we want to thin a two-dimensional Gaussian output.

    .. code::

        import jax

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (10_000,2))

    We need to define the log-target function.

    .. code::

        from jax.scipy.stats import multivariate_normal

        logprob_fn = multivariate_normal.logpdf

    In this simple example, we could have implemented `logprob_fn` ourselves as follows.

    .. code::

        import jax.numpy as jnp

        logprob_fn = lambda x: -0.5*jnp.sum(x**2)

    Next, we compute a lengthscale, the scores and laplacian of the log probabilities.

    .. code::

        from kernax.utils import median_heuristic
        lengthscale = median_heuristic(x)

        score_fn = lambda x: -x
        laplace_fn = lambda x: -1.0

        log_p = jax.vmap(logprob_fn, 0)(x)
        score_p = jax.vmap(score_fn, 0)(x)
        laplace_log_p = jax.vmap(laplace_fn, 0)(x)

    Note that the last two quantities can also be computed by automatic differentiation as follows.

    .. code::

        score_fn = jax.grad(logprob_fn)
        score_p = jax.vmap(score_fn, 0)(x)
        laplace_log_p = kernax.laplace_log_p_softplus(x, logprob_fn)

    We can now build the regularized Stein thinning function and call it.

    .. code::

        stein_thinning_fn = SteinThinning(x, log_p, score_p, laplace_log_p, lengthscale)

        idx = stein_thinning_fn(m = 1_000)

    Note that by default, the `RegularizedSteinThinning` uses the median heuristic for the lenthscale if none is passed.

    Parameters
    ----------
    x
        Sample matrix of size :math:`(n, d)`
    log_p
        Callable function built with JAX that returns the log-probability :math:`\log p(x)` of a point :math:`x \in \mathbb{R}^d`.
    score_p
        Optional callable function built with JAX that returns the gradient of :math:`\log p(x)`. Computed with automatic differentiation if not specified by the user.
    laplace_log_p
        Optional callable function built with JAX that returns the laplacian :math:`\Delta \log p(x)`. Computed with automatic differentiation if not specified by the user.
    lengthscale
        Scalar lengthscale that will be used for the underlying IMQ kernel

    Returns
    -------

    A callable function `stein_thinning_fn` which takes two input arguments `m` and `weight_entropy`.

    """

    def __new__(
        cls,
        x: Array,
        log_p: Array,
        score_p: Array,
        laplace_log_p: Array,
        lengthscale: Optional[float] = None
    ) -> Callable:

        if lengthscale is None:
            lengthscale = jnp.array(median_heuristic(x))  # type: ignore

        stein_fn = jax.tree_util.Partial(kernels.SteinIMQ, lengthscale=lengthscale)
        # stein_fn = kernels.GetSteinFn(jax.tree_util.Partial(kernels.IMQ, lengthscale=lengthscale))
        kpmap = jax.vmap(stein_fn, (None, None, 0, 0))

        def stein_thinning_fn(m: int, weight_entropy: float = None) -> Array:  # type: ignore
            if weight_entropy is None:
                weight_entropy = 1.0/float(m)

            def step_fn(carry, el):
                obj, idx = carry
                kp = kpmap(x[idx], score_p[idx], x, score_p)
                obj += 2.0*kp - weight_entropy*log_p
                new_idx = jnp.argmin(obj)
                return (obj, new_idx), new_idx
            init_obj = jax.vmap(lambda x1, s1: stein_fn(x1, s1, x1, s1), (0, 0))(x, score_p) + laplace_log_p - weight_entropy*log_p
            init_idx = jnp.argmin(init_obj)
            _, idx = jax.lax.scan(step_fn, (init_obj, init_idx), np.arange(1, m))
            return jnp.append(idx, init_idx)

        return stein_thinning_fn
