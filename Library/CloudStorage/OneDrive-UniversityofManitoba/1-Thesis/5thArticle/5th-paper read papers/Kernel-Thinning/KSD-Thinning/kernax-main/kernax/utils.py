# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import pdist

from typing import Callable
from kernax.types import Array


def median_heuristic(x: np.ndarray):
    """Function that computes the median heuristic for the lengthscale parameter.

    Parameters
    ----------

    x
        Sample matrix of dimensions :math:`(n \times d)`.

    Returns
    -------

    Median of the pairwise distances.
    """
    return np.median(pdist(x))


def laplace_log_p_hardplus(x: Array, logprob_fn: Callable) -> Array:
    """Function that computes the clipped laplacian of a log-probability for the provided sample matrix.

    Parameters
    ----------

    x
        Sample matrix of size :math:`(n, d)`
    logprob_fn
        Callable log-probability function

    Returns
    -------

    Values of the laplacian of log-probability.
    """
    jacobian_fn = jax.jacfwd(jax.jacrev(logprob_fn))
    jacobians = jax.vmap(jacobian_fn, 0)(x)
    llp = jnp.nan_to_num(jnp.clip(jax.vmap(jnp.trace)(jacobians),a_min=0))
    return llp


def laplace_log_p_softplus(x: Array, logprob_fn: Callable) -> Array:
    """Function that computes the sum of positive second-order derivatives of the log-probability for the provided sample matrix.

    Parameters
    ----------

    x
        Sample matrix of size :math:`(n, d)`
    logprob_fn
        Callable log-probability function

    Returns
    -------

    Values of the laplacian of log-probability.
    """
    jacobian_fn = jax.jacfwd(jax.jacrev(logprob_fn))
    jacobians = jnp.nan_to_num(jnp.clip(jax.vmap(jacobian_fn, 0)(x), a_min=0))
    llp = jax.vmap(jnp.trace)(jacobians)
    return llp


def laplace_log_p_flax(n: int, x: Array, logprob_fn: Callable) -> Array:
    """Function that computes the sum of positive second-order derivatives of the log-probability for the provided sample matrix.

    Parameters
    ----------

    x
        Sample matrix of size :math:`(n, d)`
    logprob_fn
        Callable log-probability function

    Returns
    -------

    Values of the laplacian of log-probability.
    """
    layers = x.keys()
    jacobian_fn = jax.jacfwd(jax.jacrev(logprob_fn))
    jacobians = jax.vmap(jacobian_fn, 0)(x)
    laplacian = jnp.zeros(n)
    for layer in layers:
        jkernel = jnp.nan_to_num(jnp.clip(jacobians[layer]["kernel"][layer]["kernel"], a_min=0))
        jbias = jnp.nan_to_num(jnp.clip(jacobians[layer]["bias"][layer]["bias"], a_min=0))
        laplacian += jax.vmap(lambda xx: jnp.trace(jnp.reshape(xx, (xx.shape[0]*xx.shape[1],xx.shape[2]*xx.shape[3]))))(jkernel) + jax.vmap(jnp.trace,0)(jbias)
    return laplacian