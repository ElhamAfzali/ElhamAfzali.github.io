# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import os
import jax
import jax.numpy as jnp
import numpy as np

from kernax import MMD
from tqdm import tqdm

def rand_gaussian_mixture(n: int, pi: list, mu, cov):
    x = []
    for _ in range(n):
        z_i = np.argmax(np.random.multinomial(1, pi))
        x_i = np.random.multivariate_normal(mu[z_i], cov[z_i], size=1).T
        x.append(x_i)
    return np.array(x).squeeze()

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('algorithm', "mala", 'Name of the MCMC method (mala, nuts)')
flags.DEFINE_float('step_size', None, 'Step size for the mala or nuts algorithms')
flags.DEFINE_integer('thinning_size', None, 'Thinning size')

def main(argv):

    data = jnp.load(os.path.join("mcmc_data", f"mog_{FLAGS.algorithm}_sz_{FLAGS.step_size}_m_{FLAGS.thinning_size}_d_2.npz"))
    pi, mu, cov = data["pi"], data["mu"], data["cov"]
    samples, idx, ridx = data["samples"], data["idx"], data["ridx"]

    num_reps = len(samples)
    mmd_wrt_m_st = np.zeros(num_reps)
    mmd_wrt_m_rst = np.zeros(num_reps)

    for i in tqdm(range(num_reps), leave=True, desc="reps"):

        xq, idx_st, idx_rst = samples[i], idx[i], ridx[i]
        xp = rand_gaussian_mixture(10_000, pi, mu, cov)

        xx = xq[idx_st]
        yy = xq[idx_rst]

        mmd_wrt_m_st[i] = MMD(x=xp, y=xx)
        mmd_wrt_m_rst[i] = MMD(x=xp, y=yy)

    mean_mmd_st, std_mmd_st = np.mean(mmd_wrt_m_st,axis=0), np.std(mmd_wrt_m_st,axis=0)
    mean_mmd_rst, std_mmd_rst = np.mean(mmd_wrt_m_rst,axis=0), np.std(mmd_wrt_m_rst,axis=0)

    np.savez(os.path.join("mmd_data", f"metrics_{FLAGS.algorithm}_sz_{FLAGS.step_size}_m_{FLAGS.thinning_size}.npz"),
             mean_mmd_st=mean_mmd_st, std_mmd_st=std_mmd_st, mean_mmd_rst=mean_mmd_rst, std_mmd_rst=std_mmd_rst)

if __name__ == "__main__":

    app.run(main)