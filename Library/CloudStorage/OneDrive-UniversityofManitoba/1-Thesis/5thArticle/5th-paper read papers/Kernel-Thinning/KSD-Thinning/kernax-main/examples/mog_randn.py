# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import matplotlib.pyplot as plt
try:
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
    plt.style.use('seaborn-v0_8')
    FONTSIZE=24
except:
    FONTSIZE=20

import jax
import jax.numpy as jnp
import numpy as np

import kernax
from kernax.utils import median_heuristic
from kernax.toy_mixtures import GaussianMixture

def main(prefix: str, n: int, m: int, d: int, pi: jnp.ndarray, mu: jnp.ndarray, covP: jnp.ndarray, covQ: jnp.ndarray):

    dist_p = GaussianMixture(d=d, pi=pi, mu=mu, cov=covP)
    dist_q = GaussianMixture(d=d, pi=pi, mu=mu, cov=covQ)

    logprob_fn = dist_p.logprob_fn
    score_fn = jax.grad(logprob_fn)

    np.random.seed(1)
    xQ = dist_q.rand(n)
    xP = dist_p.rand(n)
    lengthscale = jnp.array(median_heuristic(xP))

    log_p = jax.vmap(logprob_fn, 0)(xQ)
    score_p = jax.vmap(score_fn, 0)(xQ)
    laplace_log_p = kernax.laplace_log_p_softplus(xQ, logprob_fn)

    st_fn = kernax.SteinThinning(xQ, score_p, lengthscale)
    regst_fn = kernax.RegularizedSteinThinning(xQ, log_p, score_p, laplace_log_p, lengthscale)

    idx_st = st_fn(m)
    idx_regst = regst_fn(m)

    xmax = np.max([np.abs(xQ[:,0].min()), xQ[:,0].max()])
    xmin = -xmax
    ymax = np.max([np.abs(xQ[:,1].min()), xQ[:,1].max()])
    ymin = -ymax

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(xQ[:,0], xQ[:,1], linestyle="", marker="o", color="k")
    ax.plot(xQ[idx_st,0], xQ[idx_st,1], linestyle="", marker="o", color="r")
    ax.tick_params(labelsize=FONTSIZE)
    ax.set_xlabel(r"$x^{(1)}$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$x^{(2)}$", fontsize=FONTSIZE)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([xmin, xmax])
    ax.set_title(r"$\mathrm{Stein}$ $\mathrm{Thinning}$", fontsize=FONTSIZE)
    fig.savefig(f"{prefix}_st_n{n}_m{m}_d{d}.png", format="png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(xQ[:,0], xQ[:,1], linestyle="", marker="o", color="k")
    ax.plot(xQ[idx_regst,0], xQ[idx_regst,1], linestyle="", marker="o", color="r")
    ax.tick_params(labelsize=FONTSIZE)
    ax.set_xlabel(r"$x^{(1)}$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$x^{(2)}$", fontsize=FONTSIZE)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([xmin, xmax])
    ax.set_title(r"$\mathrm{Regularized}$ $\mathrm{Stein}$ $\mathrm{Thinning}$", fontsize=FONTSIZE)
    fig.savefig(f"{prefix}_rst_n{n}_m{m}_d{d}.png", format="png", bbox_inches="tight")

    # Laplacian correction heatmap
    jac_fn = jax.jacfwd(jax.jacrev(logprob_fn))

    xx = np.linspace(xmin, xmax, 400)
    yy = np.linspace(ymin, ymax, 400)
    XX, YY = np.meshgrid(xx, yy)
    XQ = np.stack([XX.ravel(), YY.ravel()], axis = 1)

    laplacian_correction = jnp.nan_to_num(jax.vmap(jnp.trace)(jnp.clip(jax.vmap(jac_fn)(XQ),a_min=0)))

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.contourf(XX, YY, np.reshape(laplacian_correction, (400, 400)), cmap = "jet", levels = 30)
    ax.set_title(r"$\mathrm{Laplacian}$ $\mathrm{correction}$ $\mathrm{heatmap}$", fontsize=16)
    fig.savefig(f"{prefix}_laplacian_correction.png", format = "png")

    return xQ, xP, idx_st, idx_regst

if __name__ == "__main__":

    # Figures pathology I
    d = 2
    pi = jnp.array([0.2, 0.8])
    mu = jnp.array([[-3.0]+[0.0]*(d-1), [3.0]+[0.0]*(d-1)])
    covP = [jnp.eye(d)]*2
    covQ = covP

    xQ, xP, idx_st, idx_regst = main(prefix="mog2_unbalanced", n=3000, m=300, d=d, pi=pi, mu=mu, covP=covP, covQ=covQ)

    # Figures pathology II
    d = 2
    pi = jnp.array([0.5]*2)
    mu = jnp.array([[-2.0]+[0.0]*(d-1), [2.0]+[0.0]*(d-1)])
    covP = [jnp.eye(d)]*2
    covQ = covP

    xQ, xP, idx_st, idx_regst = main(prefix="mog2", n=3000, m=300, d=d, pi=pi, mu=mu, covP=covP, covQ=covQ)

    # Unbalanced four modes
    d = 2
    pi = jnp.array([0.1, 0.4, 0.1, 0.4])
    mu = jnp.array([[-3.0, 3.0],
                    [3.0, 3.0],
                    [-3.0, -3.0],
                    [3.0, -3.0]]
    )
    covP = [jnp.eye(d)]*4
    covQ = covP

    xQ, xP, idx_st, idx_regst = main(prefix="mog4", n=3000, m=300, d=d, pi=pi, mu=mu, covP=covP, covQ=covQ)
    print("Stein thinning")
    x_idx = xQ[idx_st]
    w1 = sum((x_idx[:,0]<0) & (x_idx[:,1]>0))/300
    w2 = sum((x_idx[:,0]<0) & (x_idx[:,1]<0))/300
    w3 = sum((x_idx[:,0]>0) & (x_idx[:,1]>0))/300
    w4 = sum((x_idx[:,0]>0) & (x_idx[:,1]<0))/300
    print(f"Weights:", w1, w2, w3, w4)
    print("Regularized Stein thinning")
    x_idx = xQ[idx_regst]
    w1 = sum((x_idx[:,0]<0) & (x_idx[:,1]>0))/300
    w2 = sum((x_idx[:,0]<0) & (x_idx[:,1]<0))/300
    w3 = sum((x_idx[:,0]>0) & (x_idx[:,1]>0))/300
    w4 = sum((x_idx[:,0]>0) & (x_idx[:,1]<0))/300
    print(f"Weights:", w1, w2, w3, w4)

    # Circle
    d = 2
    modes = 6
    radius = 3
    pi = jnp.ones(modes)/modes
    covP = [jnp.eye(d)]*modes
    covQ = covP
    mu = jnp.array([
        [
        np.cos(2.0*np.pi*i/modes)*radius,
        np.sin(2.0*np.pi*i/modes)*radius
        ] for i in range(modes)]
    )

    xQ, xP, idx_st, idx_regst = main(prefix="circle", n=3000, m=300, d=d, pi=pi, mu=mu, covP=covP, covQ=covQ)