# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import jax
import jax.numpy as jnp

from typing import Callable
from kernax.types import Array


class KernelQuantization:
    """Implements kernel quantization by greedily minimizing the maximum mean discrepancy.

    [1] Chen, Y., Welling, M., & Smola, A. (2012).
    Super-samples from kernel herding. arXiv preprint arXiv:1203.3472

    [2] Teymur, O., Gorham, J., Riabiz, M., & Oates, C. (2021, March).
    Optimal quantisation of probability measures using maximum mean discrepancy. In International Conference on Artificial Intelligence and Statistics (pp. 1027-1035). PMLR.

    Examples
    --------

    Assume that we want to summarize a two-dimensional Gaussian sample of size 10,000.

    .. code::

        import jax
        import jax.random as jr

        key = jr.PRNGKey(0)
        X = jr.normal(key, (10_000, 2))

    Now that we have a toy sample, let's pick a kernel function.
    Here, we use the energy kernel function. In this case, the maximum mean discrepancy reduces to the energy distance.

    .. code::

        from kernax.kernels import Energy
        from kernax import KernelQuantization

        quant_fn = KernelQuantization(X, kernel_fn=Energy)
        idx = quant_fn(m = 1_000)

    The output `idx` gathers the selected indices. Let's plot the result.

    .. code::

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X[:, 0], X[:, 1], ls="", marker="o", color="k", label="Initial sample")
        ax.plot(X[idx, 0], X[idx, 1], ls="", marker="o", color="r", label="Selected sample")
        ax.legend(fontsize=14)

    """

    def __new__(
        cls,
        x: Array,
        kernel_fn: Callable,
    ) -> Callable:

        def gram_matrix_fn(X: Array, Y: Array):
            return jax.vmap(lambda xx: jax.vmap(lambda yy: kernel_fn(xx, yy))(Y))(X)

        k0_diag = jax.vmap(lambda xx: kernel_fn(xx, xx))
        kmap = jax.jit(jax.vmap(kernel_fn, (None, 0)))

        def mmd_quantization_fn(m: int) -> Array:
            Kmat = gram_matrix_fn(x, x)
            k0_mean = jnp.mean(Kmat, axis=1)
            k0 = k0_diag(x)
            obj = k0 - 2.0*k0_mean
            init = jnp.argmin(obj)

            def thinning_step_fn(carry, xs):
                idx, obj = carry
                ki = kmap(x[idx], x)
                obj = obj + 2.0*ki - 2.0*k0_mean
                new_idx = jnp.argmin(obj)
                return (new_idx, obj), new_idx

            _, idx = jax.lax.scan(thinning_step_fn, (init, obj), jnp.arange(1, m))
            return jnp.append(init, idx)

        return mmd_quantization_fn
