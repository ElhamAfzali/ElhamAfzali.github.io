# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from scipy.stats import multivariate_t

from typing import NamedTuple, Callable, List
from functools import partial

def _rand_gaussian_mixture(n: int, pi: list, mu, cov):
    x = []
    for _ in range(n):
        z_i = np.argmax(np.random.multinomial(1, pi))
        x_i = np.random.multivariate_normal(mu[z_i], cov[z_i], size=1).T
        x.append(x_i)
    return jnp.array(x).squeeze()

def _rand_gaussian_banana(mu: np.ndarray, cov: np.ndarray):
    d = len(mu)
    b = 0.025
    z = np.random.multivariate_normal(np.zeros(d), np.diag(cov), size=1).T
    x = np.zeros_like(z)
    x = z
    x[1] += b*z[0]**2 - 100*b
    y = x.squeeze() + mu
    return y

def _rand_gaussian_banana_mixture(mu: np.ndarray, pi: np.ndarray, cov_diag: np.ndarray, size: int=10_000):
    _, d = mu.shape
    x = np.zeros((size,d))
    for i in range(size):
        zi = np.argmax(np.random.multinomial(1, pi))
        x[i] = _rand_gaussian_banana(mu[zi], cov_diag)
    return x

def _rand_t_banana(mu: np.ndarray, cov: np.ndarray, df: int):
    d = len(mu)
    b = 0.025
    z = multivariate_t.rvs(np.zeros(d), cov, df=df, size=1)
    x = np.zeros_like(z)
    x = z
    x[1] += b*z[0]**2 - 100*b
    return x + mu

def _rand_t_banana_mixture(mu: np.ndarray, pi: np.ndarray, cov_diag: np.ndarray, df: int, size: int=10_000):
    _, d = mu.shape
    x = np.zeros((size,d))
    for i in range(size):
        zi = np.argmax(np.random.multinomial(1, pi))
        x[i] = _rand_t_banana(mu[zi], np.diag(cov_diag), df)
    return x

def _logpdf_multivariate_t(x, loc, df, prec_U, dim):
    dev = x - loc
    maha = jnp.sum(jnp.square(dev*prec_U))
    t = 0.5 * (df + dim)
    logpdf = -t * jnp.log(1. + (1./df) * maha) + 2*dim
    return logpdf

def _pdf_multivariate_t(x, loc, df, prec_U, dim):
    log_pdf = _logpdf_multivariate_t(x, loc, df, prec_U, dim)
    return jnp.exp(log_pdf)

class Distribution(NamedTuple):
    d: int
    logprob_fn: Callable
    rand: Callable

class GaussianMixture:

    def __new__(
        cls,
        d: int,
        pi: List[int],
        mu: jnp.ndarray,
        cov: jnp.ndarray) -> Distribution:

        def logprob_fn(x: jnp.ndarray):
            K = 1.5*d
            pdf = pi[0]*jnp.exp(K + multivariate_normal.logpdf(x, mu[0], cov[0]))
            for i in range(1,len(pi)):
                pdf += pi[i]*jnp.exp(K + multivariate_normal.logpdf(x, mu[i], cov[i]))
            return jnp.log(pdf)

        rand = partial(_rand_gaussian_mixture, pi=pi, mu=mu, cov=cov)
        return Distribution(d=d, logprob_fn=logprob_fn, rand=rand)

class gBananaMixture:
    """Gaussian banana mixture.
    """

    def __new__(
        cls,
        d: int,
        pi: List[int],
        mu: List[np.ndarray]
        ) -> Distribution:

        dim = d
        cov_diag = np.array([30] + [1]*(d-1))

        def logprob_fn(x: np.ndarray):
            b = 0.025
            z = x-mu[0]
            z = z.at[1].set(x[1]-b*z[0]**2 + 100*b - mu[0][1])
            pdf = pi[0]*multivariate_normal.pdf(z, jnp.zeros(dim), jnp.diag(cov_diag))
            for i in range(1,len(pi)):
                z = x-mu[i]
                z = z.at[1].set(x[1]-b*z[0]**2 + 100*b - mu[i][1])
                pdf += pi[i]*multivariate_normal.pdf(z, jnp.zeros(dim), jnp.diag(cov_diag))
            return jnp.log(pdf)

        rand = partial(_rand_gaussian_banana_mixture, mu=mu, pi=pi, cov_diag=cov_diag)
        return Distribution(d=d, logprob_fn=logprob_fn, rand=rand)

class tBananaMixture:
    """Student-t banana mixture.
    """

    def __new__(
        cls,
        d: int,
        pi: List[int],
        mu: List[np.ndarray],
        df: int) -> Distribution:

        dim = d
        cov_diag = np.array([30] + [1]*(d-1))
        prec_U = jnp.sqrt(1.0/cov_diag)

        def logprob_fn(x: np.ndarray):
            b = 0.025
            z = x-mu[0]
            z = z.at[1].set(x[1]-b*z[0]**2 + 100*b - mu[0][1])
            pdf = pi[0]*_pdf_multivariate_t(z, jnp.zeros(dim), df, prec_U, dim)
            for i in range(1,len(pi)):
                z = x-mu[i]
                z = z.at[1].set(x[1]-b*z[0]**2 + 100*b - mu[i][1])
                pdf += pi[i]*_pdf_multivariate_t(z, jnp.zeros(dim), df, prec_U, dim)
            return jnp.log(pdf)

        rand = partial(_rand_t_banana_mixture, mu=mu, pi=pi, cov_diag=cov_diag, df=df)
        return Distribution(d=d, logprob_fn=logprob_fn, rand=rand)