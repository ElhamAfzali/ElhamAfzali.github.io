# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import jax
import jax.numpy as jnp
from kernax.kernels import Energy, IMQ, GetSteinFn
from kernax.utils import median_heuristic
from typing import Callable
from kernax.types import Array

def MMD(x: Array, y: Array) -> float:
    """Implements the V-estimator the maximum mean discrepancy.

    Maximum mean discrepancy between two probability distributions :math:`P` and :math:`Q` is defined as:

    .. math::

        \mathrm{MMD}^2(P,Q) = \mathbb{E}_{x\sim P}\mathbb{E}_{y\sim P}k(x,y) + \mathbb{E}_{x\sim Q}\mathbb{E}_{y\sim Q}k(x,y) - 2 \mathbb{E}_{x\sim P}\mathbb{E}_{y\sim Q}k(x,y)

    Given two samples :math:`\{x_i\}_{i=1}^{n}` and :math:`\{y_i\}_{i=1}^{m}`, this function estimates MMD with a V-statistics.

    Examples
    --------

    Assume that we have two samples from Gaussian distributions.

    .. code::

        import jax

        key = jax.random.PRNGKey(0)
        rng_key1, rng_key2 = jax.random.split(key)

        x = jax.random.normal(rng_key1, (10_000,2))
        y = jax.random.normal(rng_key2, (10_000,2)) + jnp.array([0.1, 0.5])[:,None]


    The maximum mean discrepancy estimated by a V-statistics can be obtained as follows.

    .. code::

        mmd = MMD(x, y)

    Parameters
    ----------

    x
        Sample matrix of size :math:`(n, d)`
    y
        Sample matrix of size :math:`(n, d)`

    Returns
    -------

    The V-estimator of the maximum mean discrepancy between the provided samples.
    """

    kxx = jax.vmap(lambda x1: jax.vmap(lambda y1: Energy(x1, y1))(x))(x)
    kyy = jax.vmap(lambda x1: jax.vmap(lambda y1: Energy(x1, y1))(y))(y)
    kxy = jax.vmap(lambda x1: jax.vmap(lambda y1: Energy(x1, y1))(x))(y)
    mmd = (1.0/x.shape[0]**2)*kxx.sum() + (1.0/y.shape[0]**2)*kyy.sum() - (2.0/(x.shape[0]*y.shape[0]))*kxy.sum()
    return jnp.sqrt(mmd)

def KSD(x: Array, sx: Array, kernel_fn: Callable=None) -> float:
    """Implements the V-estimator of kernelized Stein discrepancy.

    Kernelized Stein discrepancy between two probability distributions is defined as:

    .. math::
        \mathrm{KSD}^2(P,Q) = E_{x\sim Q}E_{y\sim Q} k_p(x,y)

    where :math:`k_p` denotes the Stein kernel.

    Examples
    --------

    Assume that we have a sample from a Gaussian distribution.

    .. code::

        import jax

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (10_000,2)) + jnp.array([1.0, 1.0])[None,:]

    The target distrubution is also a Gaussian distribution but centered. Hence, the scores are obtained as follows.

    .. code::

        score_fn = lambda x: -x
        sx = score_fn(x)

    We can now compute KSD.

    .. code::

        ksd = KSD(x, sx)

    Note that we can also specify the underlying kernel involved in the Stein kernel.

    This function uses the inverse multi-quadratric kernel by default but any other kernel function can be passed as an argument.

    .. code::

        from kernax.kernels import Gaussian

        ksd = KSD(x, sx, Gaussian)

    The lengthscale (or bandwidth) of the underyling `kernel_fn` is automatically set to the median heuristic.

    Parameters
    ----------

    x
        Sample matrix of size :math:`(n, d)`
    sx
        Matrix of size :math:`(n, d)` containing the scores of the sample :math:`x`.
    kernel_fn
        Callable corresponding the the underlying kernel function in the Stein kernel (Default: IMQ kernel).

    Returns
    -------

    The V-estimator of kernelized Stein discrepancy given the provided sample and chosen score function.
    """

    if kernel_fn is None:
        lengthscale = jnp.array(median_heuristic(x))
        kernel_fn = jax.tree_util.Partial(IMQ, lengthscale=lengthscale)

    kp_fn = GetSteinFn(kernel_fn)
    kp = jax.vmap(lambda a, b: jax.vmap(lambda c, d: kp_fn(a,b,c,d))(x,sx))(x,sx)
    ksd = jnp.sqrt(jnp.sum(kp))/x.shape[0]
    return ksd