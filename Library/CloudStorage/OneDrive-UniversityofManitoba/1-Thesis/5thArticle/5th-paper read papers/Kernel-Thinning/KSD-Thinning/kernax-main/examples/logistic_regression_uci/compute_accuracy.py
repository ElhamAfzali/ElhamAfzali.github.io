# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import os
import jax.numpy as jnp

if __name__ == "__main__":

    algorithms = ["mala"]
    datasets = ["BreastCancer", "Diabetes", "HabermanSurvival", "LiverDisorder", "Sonar"]
    step_sizes = [0.01, 0.001, 0.0001, 1e-05]
    thinning_sizes = [50, 100, 300]

    for algorithm in algorithms:
        auc_mean_st = {}

        for dataset in datasets:
            auc_mean_st[dataset] = {}

            for thinning_size in thinning_sizes:
                mean_auc_st, mean_auc_rst = [], []
                std_auc_st, std_auc_rst = [], []
                for step_size in step_sizes:
                    filename = os.path.join("mcmc_data", f"logistic_regression_{dataset}_{algorithm}_sz_{step_size}_m_{thinning_size}.npz")
                    if os.path.isfile(filename):
                        # data = jnp.load(os.path.join("22_11_2022", filename))
                        data = jnp.load(filename)

                        mean_auc_st += [data["mean_auc_st"]]
                        mean_auc_rst += [data["mean_auc_rst"]]
                        std_auc_st += [data["std_auc_st"]]
                        std_auc_rst += [data["std_auc_rst"]]
                    else:
                        raise ValueError(f"Missing {filename}")

                ist = jnp.argmax(jnp.array(mean_auc_st))
                irst = jnp.argmax(jnp.array(mean_auc_rst))
                auc_mean_st[dataset][str(thinning_size)] = "{:1.2f} ({:1.3f}) {:1.2f} ({:1.3f})".format(round(mean_auc_st[ist].item(),2), round(std_auc_st[ist].item(),4), round(mean_auc_rst[ist].item(),2), round(std_auc_rst[ist].item(),4))

        import pandas as pd
        df = pd.DataFrame.from_dict(auc_mean_st)
        print(df.T)
        print("*"*100)