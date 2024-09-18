# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import os
import jax
import jax.numpy as jnp
import numpy as np

from kernax import MMD
from tqdm import tqdm

from kernax.toy_mixtures import tBananaMixture
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('algorithm', "mala", 'Name of the MCMC method (mala, nuts)')
flags.DEFINE_float('step_size', None, 'Step size for the mala or nuts algorithms')
flags.DEFINE_integer('dimension', 2, 'Dimension of the Gaussian mixture')

def main(argv):

    data = jnp.load(os.path.join("mcmc_data", f"mob_{FLAGS.algorithm}_sz_{FLAGS.step_size}_m_300_d_{FLAGS.dimension}.npz"))
    pi, mu, df = data["pi"], data["mu"], data["df"]
    samples, idx, ridx = data["samples"], data["idx"], data["ridx"]

    num_reps = len(samples)
    mmd_wrt_m_st = np.zeros(num_reps)
    mmd_wrt_m_rst = np.zeros(num_reps)

    dist_p = tBananaMixture(d=FLAGS.dimension, pi=pi, mu=mu, df=df)

    for i in tqdm(range(num_reps), leave=True, desc="reps"):

        xq, idx_st, idx_rst = samples[i], idx[i], ridx[i]
        xp = dist_p.rand()

        xx = xq[idx_st]
        yy = xq[idx_rst]

        mmd_wrt_m_st[i] = MMD(x=xp, y=xx)
        mmd_wrt_m_rst[i] = MMD(x=xp, y=yy)

    mean_mmd_st, std_mmd_st = np.mean(mmd_wrt_m_st,axis=0), np.std(mmd_wrt_m_st,axis=0)
    mean_mmd_rst, std_mmd_rst = np.mean(mmd_wrt_m_rst,axis=0), np.std(mmd_wrt_m_rst,axis=0)

    np.savez(os.path.join("mmd_data", f"metrics_{FLAGS.algorithm}_sz_{FLAGS.step_size}_m_300_d_{FLAGS.dimension}.npz"),
             mean_mmd_st=mean_mmd_st, std_mmd_st=std_mmd_st, mean_mmd_rst=mean_mmd_rst, std_mmd_rst=std_mmd_rst)

if __name__ == "__main__":

    app.run(main)