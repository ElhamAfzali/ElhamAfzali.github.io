# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import os
import jax.numpy as jnp
import numpy as np

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

def rand_gaussian_mixture(n: int, pi: list, mu, cov):
    x = []
    for _ in range(n):
        z_i = np.argmax(np.random.multinomial(1, pi))
        x_i = np.random.multivariate_normal(mu[z_i], cov[z_i], size=1).T
        x.append(x_i)
    return np.array(x).squeeze()

if __name__ == "__main__":

    thinning_sizes = [100, 300, 500, 700, 900, 1100]
    nm = len(thinning_sizes)

    for ai, algorithm in enumerate(["mala", "nuts"]):

        subfig, axs = plt.subplots(1, 3, figsize=(3*6,4), sharex=True, sharey="row")
        markers = ["o", "s"]
        labels = [r"$\mathrm{Stein}$ $\mathrm{Thinning}$", r"$\mathrm{Regularized}$ $\mathrm{Stein}$ $\mathrm{Thinning}$"]

        for si, step_size in enumerate(["0.05", "0.1", "0.5"]):

            mean_mmd_st, std_mmd_st = np.zeros(nm), np.zeros(nm)
            mean_mmd_rst, std_mmd_rst = np.zeros(nm), np.zeros(nm)

            for i, thinning_size in enumerate(thinning_sizes):
                data = jnp.load(os.path.join("mmd_data",f"metrics_{algorithm}_sz_{step_size}_m_{thinning_size}.npz"))

                mean_mmd_st[i] = data["mean_mmd_st"]
                std_mmd_st[i] = data["std_mmd_st"]

                mean_mmd_rst[i] = data["mean_mmd_rst"]
                std_mmd_rst[i] = data["std_mmd_rst"]

            fig, ax = plt.subplots(1, 1, figsize=(6,4))
            ax.plot(thinning_sizes, mean_mmd_st, linewidth=2, linestyle="-", color="b", marker="o", label=labels[0])
            ax.plot(thinning_sizes, mean_mmd_rst, linewidth=2, linestyle="-", color="r", marker="o", label=labels[1])
            ax.fill_between(thinning_sizes, mean_mmd_st-std_mmd_st, mean_mmd_st+std_mmd_st, color="tab:blue", alpha=0.25)
            ax.fill_between(thinning_sizes, mean_mmd_rst-std_mmd_rst, mean_mmd_rst+std_mmd_rst, color="tab:red", alpha=0.25)
            ax.set_xlabel(r"$m$", fontsize=24)
            ax.set_ylabel(r"$\mathrm{MMD}(P,Q_m)$", fontsize=24)
            ax.tick_params(labelsize=22)
            ax.set_xticks(thinning_sizes)
            ax.legend(fontsize=22)
            fig.savefig(f"mog_{algorithm}_sz_{step_size}_thinning_size.png", format="png", bbox_inches="tight")

            axs[si].plot(thinning_sizes, mean_mmd_st, linewidth=2, linestyle="-", color="b", marker="o", label=labels[0])
            axs[si].plot(thinning_sizes, mean_mmd_rst, linewidth=2, linestyle="-", color="r", marker="o", label=labels[1])
            axs[si].fill_between(thinning_sizes, mean_mmd_st-std_mmd_st, mean_mmd_st+std_mmd_st, color="tab:blue", alpha=0.25)
            axs[si].fill_between(thinning_sizes, mean_mmd_rst-std_mmd_rst, mean_mmd_rst+std_mmd_rst, color="tab:red", alpha=0.25)
            axs[si].set_xlabel(r"$m$", fontsize=20)
            axs[si].set_ylabel(r"$\mathrm{MMD}(P,Q_m)$", fontsize=20)
            axs[si].tick_params(labelsize=18)
            axs[si].set_xticks([0] + thinning_sizes[1:])
            axs[si].set_title(rf"$\epsilon = {step_size}$", fontsize=18)

        handles, labels = axs[0].get_legend_handles_labels()
        subfig.legend(handles, labels, loc=(0.12, 0.40), ncols=1, fontsize=20)
        subfig.savefig(f"subplot_mog_{algorithm}_thinning_size.png", format="png", bbox_inches="tight")