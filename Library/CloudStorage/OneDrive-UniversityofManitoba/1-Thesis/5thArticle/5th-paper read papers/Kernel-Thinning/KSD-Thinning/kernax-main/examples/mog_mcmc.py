# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import os
import jax
import jax.numpy as jnp

import kernax
from kernax.utils import median_heuristic

import numpy as np
from tqdm import tqdm

from absl import app
from absl import logging
from absl import flags

from kernax.toy_mixtures import GaussianMixture

FLAGS = flags.FLAGS
flags.DEFINE_string('algorithm', "mala", 'Name of the MCMC method (mala, nuts)')
flags.DEFINE_string('output_path', '.', 'Path where the results are stored')
flags.DEFINE_string('lambda_entropy', 'inverse', 'Form of the weight regularization parameter: inverse = 1/m, sq_inverse = 1/m^2, log_inverse = 1/log(m)')
flags.DEFINE_string('case', 'mog4', 'Type of Gaussian mixture (mog2, mog4). They differ in: number of modes (2 or 4) but also in mean values and covariances.')
flags.DEFINE_integer('dim', 2, 'Dimension of the Gaussian mixture')
flags.DEFINE_integer('num_iterations', 100_000, 'Number of iterations for the mala or nuts algorithm')
flags.DEFINE_integer('thinning_size', 300, 'Number of particles selected by Stein thinning')
flags.DEFINE_integer('num_repetitions', 20, 'Number of repetitions to get uncertainties')
flags.DEFINE_float('step_size', 1e-3, 'Step size for the mala or nuts algorithms')

def main(argv):
    
    d, n, m = FLAGS.dim, FLAGS.num_iterations, FLAGS.thinning_size
    num_reps = FLAGS.num_repetitions
    step_size = FLAGS.step_size
    
    if FLAGS.lambda_entropy=="inverse":
        weight_entropy = 1.0/m
    elif FLAGS.lambda_entropy=="sq_inverse":
        weight_entropy = 1.0/m**2
    elif FLAGS.lambda_entropy=="log_inverse":
        weight_entropy = 1.0/jnp.log(m)
    else:
        raise ValueError("Wrong lambda_entropy. Should be inverse, sq_inverse or log_inverse")
    
    if FLAGS.case=="mog4":
        pi = jnp.array([0.25]*4)
        cov_p = [jnp.eye(d)]*2 + [2*jnp.eye(d)]*2
        mu = [jnp.array([-2.0]+[0.0]*(d-1)), jnp.array([2.0]+[0.0]*(d-1)), jnp.array([-3.0,4.0]+[0.0]*(d-2)), jnp.array([3.0,4.0]+[0]*(d-2))]        
        dist_p = GaussianMixture(d=d, pi=pi, mu=mu, cov=cov_p)
    elif FLAGS.case=="mog2":
        pi = jnp.array([0.2, 0.8])
        cov_p = [jnp.eye(d)]*2
        mu = jnp.array([[-3.0]+[0.0]*(d-1), [3.0]+[0.0]*(d-1)])
        dist_p = GaussianMixture(d=d, pi=pi, mu=mu, cov=cov_p)
    else:
        raise ValueError("Case should be mog2 or mog4")
    
    logprob_fn = dist_p.logprob_fn
    score_fn = jax.grad(logprob_fn)
    
    if FLAGS.algorithm=="nuts":
        inverse_mass_matrix = jnp.ones(d)
        def sample_fn(init_key):
            init_positions = jax.random.normal(init_key, (d,))
            _, rng_key = jax.random.split(init_key)
            _, states = kernax.nuts(logprob_fn, init_positions, n, step_size, inverse_mass_matrix, rng_key)
            return states.position
    elif FLAGS.algorithm=="mala":
        def sample_fn(init_key):
            init_positions = jax.random.normal(init_key, (d,))
            _, rng_key = jax.random.split(init_key)            
            _, states = kernax.mala(logprob_fn, init_positions, n, step_size, rng_key)
            return states.position
    
    rng_key = jax.random.PRNGKey(np.random.randint(0,1234567))
    rep_keys = jax.random.split(rng_key, num_reps)
        
    idx, ridx = [], []
    samples = []
    for key in tqdm(rep_keys, leave=True):
        
        logging.info(f"Sampling with {FLAGS.algorithm}")
        positions = sample_fn(key)    
        
        logging.info(f"Computing log-prob, scores, and laplacians")
        log_p = jax.vmap(logprob_fn, 0)(positions)
        score_p = jax.vmap(score_fn, 0)(positions)
        laplace_log_p = kernax.laplace_log_p_softplus(positions, logprob_fn)
        lengthscale = median_heuristic(positions)
    
        logging.info("Stein thinning")
        stein_thinning_fn = kernax.SteinThinning(positions, score_p, lengthscale)
        idx += [stein_thinning_fn(m=m)]
                
        logging.info("Regularized Stein thinning")
        stein_thinning_fn = kernax.RegularizedSteinThinning(positions, log_p, score_p, laplace_log_p, lengthscale)
        ridx += [stein_thinning_fn(m=m, weight_entropy=weight_entropy)]
        
        samples += [positions]
    
    if not os.path.isdir(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)
    np.savez(os.path.join(FLAGS.output_path, f"mog_{FLAGS.algorithm}_sz_{FLAGS.step_size}_m_{FLAGS.thinning_size}_d_{FLAGS.dim}.npz"), idx=idx, ridx=ridx, samples=samples, pi=pi, mu=mu, cov=cov_p)
            
if __name__ == "__main__":
    
    app.run(main)