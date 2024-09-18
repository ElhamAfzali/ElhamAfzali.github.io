Stein thinning
==============

Let's say that we want to thin a Gaussian sample with Stein thinning [1] and regularized Stein thinning [2], and that the target distribution is also Gaussian.

First generate a toy sample.

.. code::

    import numpy as np
    x = np.random.randn(1000, 2)

In order to apply Stein thinning, we need the values of the score function and a lengthscale.
Here, the expression of the target score function is straightforward and can be implemented by hand or with, e.g., scipy.

.. code::

    from jax.scipy.stats import multivariate_normal

    def logprob_fn(x):
        return multivariate_normal(x, mean=jnp.zeros(2), cov=jnp.eye(2))

    score_fn = jax.grad(logprob_fn)
    score_values = jax.vmap(score_fn, 0)(x)

We also need a lengthscale which is chosen as the median heuristic.

.. code::

    from kernax.utils import median_heuristic
    lengthscale = median_heuristic(x)

Stein thinning can applied as follows to select 100 points amongst the original sample.

.. code::

    from kernax import SteinThinning
    stein_fn = SteinThinning(x, score_values, lengthscale)
    indices = stein_fn(100)

Regularized Stein thinning can be used in a similar fashion but requires two additional inputs: the log-probability values,
and the regularization terms.

.. code::

    from kernax import laplace_log_p_softplus
    log_p_values = jax.vmap(logprob_fn, 0)(x)
    laplace_log_p_values = laplace_log_p_softplus(x, score_fn)

    from kernax import RegularizedSteinThinning
    reg_stein_fn = RegularizedSteinThinning(x, log_p_values, score_values, laplace_log_p_values, lengthscale)
    indices = reg_stein_fn(100)

Additional comments or examples can be found in the API documentation.

References
~~~~~~~~~~

[1] Riabiz, M., Chen, W. Y., Cockayne, J., Swietach, P., Niederer, S. A., Mackey, L., & Oates, C. J. (2022).
Optimal thinning of MCMC output. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(4), 1059-1081.

[2] Bénard, C., Staber, B., & Da Veiga, S. (2023).
Kernel Stein Discrepancy thinning: a theoretical perspective of pathologies and a practical fix with regularization. NeurIPS 2023.