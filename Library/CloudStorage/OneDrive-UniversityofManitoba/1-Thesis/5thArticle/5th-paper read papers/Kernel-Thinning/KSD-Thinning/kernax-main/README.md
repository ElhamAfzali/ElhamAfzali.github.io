<h1 align="center">Kernax: regularized Stein thinning</h1>

```python
import jax
import jax.numpy as jnp
rng_key = jax.random.PRNGKey(0)
x = jax.random.normal(rng_key, (1000,2))

from jax.scipy.stats import multivariate_normal
def logprob_fn(x):
    return multivariate_normal.logpdf(x, mean=jnp.zeros(2), cov=jnp.eye(2))
score_fn = jax.grad(logprob_fn)

score_values = jax.vmap(score_fn, 0)(x)

from kernax.utils import median_heuristic
lengthscale = jnp.array([median_heuristic(x)])

from kernax import SteinThinning
stein_fn = SteinThinning(x, score_values, lengthscale)
indices = stein_fn(100)

from kernax import laplace_log_p_softplus
log_p = jax.vmap(score_fn, 0)(x)
laplace_log_p_values = laplace_log_p_softplus(x, score_fn)

from kernax import RegularizedSteinThinning
reg_stein_fn = RegularizedSteinThinning(x, log_p, score_values, laplace_log_p_values, lengthscale)
indices = reg_stein_fn(100)
```

## Documentation

Documentation is available at [readthedocs](https://kernax.readthedocs.io/en/latest/?kernax=latest).

## Contributing

This code is not meant to be an evolving library. However, feel free to create issues and merge requests.

## Install guide

Conda is the preferred way to install kernax.
### Conda
A package is available on the conda-forge channel.
```
conda install -c conda-forge kernax
```

### PyPI
Alternatively, a pypi package is also available.
```console
pip install kernax
```

### From source
To install from source, clone this repository, then install it in editable mode:
```console
pip install -e .
```

## Reproductibility

This code implements the regularized Stein thinning algorithm introduced in the paper [Kernel Stein Discrepancy thinning: a theoretical perspective of pathologies and a practical fix with regularization](https://arxiv.org/pdf/2301.13528.pdf).

Please consider citing the paper when using this library:
```bibtex
@article{benard2023kernel,
  title={Kernel Stein Discrepancy thinning: a theoretical perspective of pathologies and a practical fix with regularization},
  author={B{\'e}nard, Cl{\'e}ment and Staber, Brian and Da Veiga, S{\'e}bastien},
  journal={arXiv preprint arXiv:2301.13528},
  year={2023}
}
```

All the numerical experiments presented in the [paper](https://arxiv.org/pdf/2301.13528.pdf) can be reproduced with the scripts made available in the example folder.

In particular:

* Figures 1, 2 & 3 can be reproduced with the script example/mog_randn.py

* Each experiment in Section 4 and Appendix 1 can be reproduced with the scripts gathered in the following folders:
    * Gaussian mixture: example/mog4_mcmc and example/mog4_mcmc_dim
    * Mixture of banana-shaped distributions: example/mobt2_mcmc and example/mobt2_mcmc_dim
    * Bayesian logistic regression: example/logistic_regression.py

* Two additional scripts are also available to reproduce figures shown in the supplementary material:
    * Figure 2: example/mog_weight_weights.py
    * Figure 6: example/mog4_mcmc_lambda