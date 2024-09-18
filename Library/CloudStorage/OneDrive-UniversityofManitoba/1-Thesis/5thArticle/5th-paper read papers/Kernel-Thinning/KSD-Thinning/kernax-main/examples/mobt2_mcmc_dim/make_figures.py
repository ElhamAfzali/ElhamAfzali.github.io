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

if __name__ == "__main__":

    dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    nd = len(dimensions)

    for ai, algorithm in enumerate(["mala", "nuts"]):

        subfig, axs = plt.subplots(1, 3, figsize=(3*6,4), sharex=True, sharey="row")
        markers = ["o", "s"]
        labels = [r"$\mathrm{Stein}$ $\mathrm{Thinning}$", r"$\mathrm{Regularized}$ $\mathrm{Stein}$ $\mathrm{Thinning}$"]

        for si, step_size in enumerate(["0.1", "0.5", "1.0"]):

            mean_mmd_st, std_mmd_st = np.zeros(nd), np.zeros(nd)
            mean_mmd_rst, std_mmd_rst = np.zeros(nd), np.zeros(nd)

            for i, dimension in enumerate(dimensions):
                data = jnp.load(os.path.join("mmd_data", f"metrics_{algorithm}_sz_{step_size}_m_300_d_{dimension}.npz"))

                mean_mmd_st[i] = data["mean_mmd_st"]
                std_mmd_st[i] = data["std_mmd_st"]

                mean_mmd_rst[i] = data["mean_mmd_rst"]
                std_mmd_rst[i] = data["std_mmd_rst"]

            fig, ax = plt.subplots(1, 1, figsize=(6,4))
            ax.plot(dimensions, mean_mmd_st, linewidth=2, linestyle="-", color="b", marker="o", label=labels[0])
            ax.plot(dimensions, mean_mmd_rst, linewidth=2, linestyle="-", color="r", marker="o", label=labels[1])
            ax.fill_between(dimensions, mean_mmd_st-std_mmd_st, mean_mmd_st+std_mmd_st, color="tab:blue", alpha=0.25)
            ax.fill_between(dimensions, mean_mmd_rst-std_mmd_rst, mean_mmd_rst+std_mmd_rst, color="tab:red", alpha=0.25)
            ax.set_xlabel(r"$d$", fontsize=24)
            ax.set_ylabel(r"$\mathrm{MMD}(P,Q_m)$", fontsize=24)
            ax.tick_params(labelsize=22)
            ax.set_xticks(dimensions)
            # ax.set_title(rf"$\epsilon = {step_size}$", fontsize=18)
            fig.savefig(f"mobt_{algorithm}_sz_{step_size}_dimension.png", format="png", bbox_inches="tight")

            axs[si].plot(dimensions, mean_mmd_st, linewidth=2, linestyle="-", color="b", marker="o", label=labels[0])
            axs[si].plot(dimensions, mean_mmd_rst, linewidth=2, linestyle="-", color="r", marker="o", label=labels[1])
            axs[si].fill_between(dimensions, mean_mmd_st-std_mmd_st, mean_mmd_st+std_mmd_st, color="tab:blue", alpha=0.25)
            axs[si].fill_between(dimensions, mean_mmd_rst-std_mmd_rst, mean_mmd_rst+std_mmd_rst, color="tab:red", alpha=0.25)
            axs[si].set_xlabel(r"$d$", fontsize=20)
            axs[si].set_ylabel(r"$\mathrm{MMD}(P,Q_m)$", fontsize=20)
            axs[si].tick_params(labelsize=16)
            axs[si].set_xticks(dimensions)
            axs[si].set_title(rf"$\epsilon = {step_size}$", fontsize=18)

        handles, labels = axs[0].get_legend_handles_labels()
        subfig.legend(handles, labels, loc=(0.05, 0.7), ncols=1, fontsize=18)
        subfig.savefig(f"subplot_mobt_{algorithm}_dimension.png", format="png", bbox_inches="tight")