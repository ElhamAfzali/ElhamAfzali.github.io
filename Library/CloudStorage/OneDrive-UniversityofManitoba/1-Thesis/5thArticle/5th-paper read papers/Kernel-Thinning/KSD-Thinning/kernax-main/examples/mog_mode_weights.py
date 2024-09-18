# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import jax
import jax.numpy as jnp
import numpy as np
import kernax
from kernax.utils import median_heuristic
from kernax.toy_mixtures import GaussianMixture
from tqdm import tqdm

def main(m: int, d: int, pi: jnp.ndarray, mu: jnp.ndarray, covP: jnp.ndarray, covQ: jnp.ndarray):

    dist_p = GaussianMixture(d=d, pi=pi, mu=mu, cov=covP)
    dist_q = GaussianMixture(d=d, pi=pi, mu=mu, cov=covQ)

    logprob_fn = dist_p.logprob_fn
    score_fn = jax.grad(logprob_fn)

    np.random.seed(1)
    xQ = dist_q.rand()
    xP = dist_p.rand()
    lengthscale = jnp.array(median_heuristic(xP))

    log_p = jax.vmap(logprob_fn, 0)(xQ)
    score_p = jax.vmap(score_fn, 0)(xQ)
    laplace_log_p = kernax.laplace_log_p_softplus(xQ, logprob_fn)

    st_fn = kernax.SteinThinning(xQ, score_p, lengthscale)
    regst_fn = kernax.RegularizedSteinThinning(xQ, log_p, score_p, laplace_log_p, lengthscale)

    idx_st = st_fn(m)
    idx_regst = regst_fn(m)

    return xQ, xP, idx_st, idx_regst

if __name__ == "__main__":

    # Quantify Pathology I
    d = 2
    pi = jnp.array([0.2, 0.8])
    mu = jnp.array([[-3.0]+[0.0]*(d-1), [3.0]+[0.0]*(d-1)])
    covP = [jnp.eye(d)]*2
    covQ = [jnp.eye(d)]*2

    n = 3000
    m = 300
    w_list = list()
    w_list_rst = list()
    for iter in tqdm(range(100)):
        xQ, xP, idx_st, idx_regst = main(m=m, d=d, pi=pi, mu=mu, covP=covP, covQ=covQ)
        w = sum(xQ[idx_st, 0] < 0)/m
        w_list.append(w)
        w_rst = sum(xQ[idx_regst, 0] < 0)/m
        w_list_rst.append(w_rst)

    # print result for standard Stein thinning
    w_mean = np.mean(w_list)
    w_sd = np.std(w_list)
    print(f"Standard - Left mode weight = {w_mean}, and right mode weight = {1 - w_mean}")
    print(f"Standard - Weight standard deviation = {w_sd}")

    # print result for regularized Stein thinning
    w_mean = np.mean(w_list_rst)
    w_sd = np.std(w_list_rst)
    print(f"Regularized - Left mode weight = {w_mean}, and right mode weight = {1 - w_mean}")
    print(f"Regularized - Weight standard deviation = {w_sd}")