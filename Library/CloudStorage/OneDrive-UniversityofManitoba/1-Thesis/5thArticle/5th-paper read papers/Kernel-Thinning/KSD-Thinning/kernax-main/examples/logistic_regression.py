# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import os
import jax
import jax.numpy as jnp
import numpy as np
import kernax
from kernax.utils import median_heuristic, laplace_log_p_softplus

from absl import app
from absl import flags
from absl import logging
from uci_datasets import BreastCancer, Ionosphere, Sonar, Authentification, HabermanSurvival, LiverDisorder, Diabetes
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', None, 'Name of the dataset (Ionosphere, Sonar, Authentification, BreastCancer)')
flags.DEFINE_string('path_to_data', None, 'Path to the folder where the data is stored')
flags.DEFINE_string('algorithm', None, 'Name of the MCMC method (mala, nuts)')
flags.DEFINE_float('step_size', None, 'Step size for the mala or nuts algorithms')
flags.DEFINE_integer('num_iterations', 1000, 'Number of iterations for the mala or nuts algorithm')
flags.DEFINE_integer('thinning_size', 300, 'Number of particles selected by Stein thinning')
flags.DEFINE_integer('num_repetitions', 10, 'Number of repitions to get uncertainties')
flags.DEFINE_string('output_path', ".", 'Folder where the outputs are dumped to')

def logprior_fn(params):
    """t-Student log prior function"""
    a, b = 1.0, 1.0
    c = (2.0*a + 1.0)/2.0
    logprobs = c*jnp.log(1.0 + jnp.square(params)/(2.0*b))
    return -jnp.sum(logprobs)

def loglikelihood_fn(params, data):
    """Log likelihood function for a given observation"""
    x, y = data
    logit = jnp.dot(x, params)
    sigmoid = jax.nn.sigmoid(logit)
    # return y*jnp.log(sigmoid) + (1-y)*jnp.log(1-sigmoid)
    return y*sigmoid - jnp.log1p(jnp.exp(sigmoid))

def predict_fn(w, x):
    """Predict function for logistic regression for a given weight w and a given input x"""
    logits = jnp.dot(w, x)
    return jax.nn.sigmoid(logits)
    
def init_parameters(rng_key, X_train):
    w = jax.random.normal(rng_key, (X_train.shape[1],))
    return w    
    
def logprob_fn(parameters, data):
    X_train, y_train = data
    logprior = logprior_fn(parameters)
    batch_loglikelihood = jax.vmap(loglikelihood_fn, (None, 0))(parameters, (X_train, y_train))
    return logprior + jnp.sum(batch_loglikelihood)

def sample_mala(rng_key, step_size, num_samples, X_train, y_train):
    # Get data and log-posterior
    log_posterior_fn = jax.tree_util.Partial(logprob_fn, data=(X_train, y_train))    

    # Initialize parameters
    init_positions = init_parameters(rng_key, X_train)

    # Run MALA
    _, rng_key = jax.random.split(rng_key)
    _, states = kernax.mala(log_posterior_fn, init_positions, num_samples, step_size, rng_key)
    return states.position
    
def sample_nuts(rng_key, step_size, num_samples, X_train, y_train):
    # Get data and log-posterior
    log_posterior_fn = jax.tree_util.Partial(logprob_fn, data=(X_train, y_train))
    
    # Initialize parameters
    inverse_mass_matrix = jnp.ones(X_train.shape[1])
    init_positions = init_parameters(rng_key, X_train)

    # Run NUTS
    _, rng_key = jax.random.split(rng_key)
    _, states = kernax.nuts(log_posterior_fn, init_positions, num_samples, step_size, inverse_mass_matrix, rng_key)
    return states.position

def main(argv):
    
    logging.info(f"Number of devices: {len(jax.devices())}")
    num_chains = len(jax.devices())
    
    step_size = FLAGS.step_size
    num_iterations = FLAGS.num_iterations
    m = FLAGS.thinning_size
    
    if FLAGS.dataset_name=="Ionosphere":
        X, y = Ionosphere(FLAGS.path_to_data)
    elif FLAGS.dataset_name=="Authentification":
        X, y = Authentification(FLAGS.path_to_data)
    elif FLAGS.dataset_name=="Sonar":
        X, y = Sonar(FLAGS.path_to_data)
    elif FLAGS.dataset_name=="BreastCancer":
        X, y = BreastCancer()
    elif FLAGS.dataset_name=="HabermanSurvival":
        X, y = HabermanSurvival(FLAGS.path_to_data)
    elif FLAGS.dataset_name=="LiverDisorder":
        X, y = LiverDisorder(FLAGS.path_to_data)
    elif FLAGS.dataset_name=="Diabetes":
        X, y = Diabetes(FLAGS.path_to_data)
    else:
        raise ValueError(f"Wrong dataset name, I got {FLAGS.dataset_name}")    
    
    logging.info(f"Dataset: {FLAGS.dataset_name}")
    logging.info(f"y.shape: {y.shape}")
    
    auc_st = []
    auc_rst = []
    for _ in tqdm(range(FLAGS.num_repetitions)):
        
        kf = KFold(n_splits=10, shuffle=True)
        
        mean_prob_st = []
        mean_prob_rst = []
        y_test = []
        
        for train_index, test_index in kf.split(X, y):
            
            X_train = jnp.array(X[train_index])
            y_train = jnp.array(y[train_index])
            X_test = jnp.array(X[test_index])
            y_test += [y[test_index]]
                
            score_fn = jax.grad(logprob_fn, argnums=0)
            
            logging.info(f"Sampling with {FLAGS.algorithm}")
            if FLAGS.algorithm=="mala":
                filename = f"logistic_regression_{FLAGS.dataset_name}_mala_sz_{step_size}_m_{FLAGS.thinning_size}.npz"
                
                pmap_mcmc = jax.pmap(sample_mala, in_axes=(0, None, None, None, None), out_axes=(0), static_broadcasted_argnums=(2,))
                rng_key = jax.random.PRNGKey(np.random.randint(0,124567))
                rng_keys = jax.random.split(rng_key, num_chains)
                position = pmap_mcmc(rng_keys, step_size, num_iterations, X_train, y_train)
                _ = jax.tree_util.tree_map(lambda x: x.block_until_ready(), position)
                position = jnp.reshape(position, (num_chains*num_iterations,-1))
                        
            elif FLAGS.algorithm=="nuts":
                filename = f"logistic_regression_{FLAGS.dataset_name}_nuts_sz_{step_size}_m_{FLAGS.thinning_size}.npz"
                
                pmap_mcmc = jax.pmap(sample_nuts, in_axes=(0, None, None, None, None), out_axes=(0), static_broadcasted_argnums=(2,))
                rng_key = jax.random.PRNGKey(np.random.randint(0,124567))
                rng_keys = jax.random.split(rng_key, num_chains)
                position = pmap_mcmc(rng_keys, step_size, num_iterations, X_train, y_train)
                _ = jax.tree_util.tree_map(lambda x: x.block_until_ready(), position)
                position = jnp.reshape(position, (num_chains*num_iterations,-1))
                
            else:
                raise ValueError("Wrong algorithm")
            
            logging.info("Computing log-probs, scores, and laplacians")
            log_p = jax.vmap(logprob_fn, (0, None))(position, (X_train, y_train))
            score_p = jax.vmap(score_fn, (0, None))(position, (X_train, y_train))
            
            log_posterior_fn = jax.tree_util.Partial(logprob_fn, data=(X_train, y_train))
            laplace_log_p = laplace_log_p_softplus(position, log_posterior_fn)
            
            subidx = np.random.randint(0,len(position),size=48_000)
            lengthscale = median_heuristic(position[subidx])
            
            logging.info("Stein thinning")
            stein_thinning_fn = kernax.SteinThinning(position, score_p, lengthscale)
            idx = stein_thinning_fn(m)
            
            logging.info("Predicting")
            params_st = position[idx]
            prob_st = jnp.stack([jax.vmap(predict_fn, in_axes=(None, 0))(params_st[i], X_test) for i in range(len(params_st))])
            mean_prob_st += [jnp.mean(prob_st, 0)]
            
            logging.info("Regularized Stein thinning")
            reg_stein_thinning_fn = kernax.RegularizedSteinThinning(position, log_p, score_p, laplace_log_p, lengthscale)
            reg_idx = reg_stein_thinning_fn(m)
            
            logging.info("Predicting")
            params_rst = position[reg_idx]
            prob_rst = jnp.stack([jax.vmap(predict_fn, in_axes=(None, 0))(params_rst[i], X_test) for i in range(len(params_rst))])
            mean_prob_rst += [jnp.mean(prob_rst, 0)]

        mean_prob_st = jnp.concatenate(mean_prob_st)
        mean_prob_rst = jnp.concatenate(mean_prob_rst)
        y_test = np.concatenate(y_test)
        
        logging.info("Computing ROC and AUC")
        fpr_st, tpr_st, _ = roc_curve(y_test, mean_prob_st, drop_intermediate=False)
        auc_st += [auc(fpr_st, tpr_st)]
        
        logging.info("Computing ROC and AUC")
        fpr_rst, tpr_rst, _ = roc_curve(y_test, mean_prob_rst, drop_intermediate=False)
        auc_rst += [auc(fpr_rst, tpr_rst)]
        
    mean_auc_st = np.mean(auc_st)
    std_auc_st = np.std(auc_st)
    
    mean_auc_rst = np.mean(auc_rst)
    std_auc_rst = np.std(auc_rst)
    
    logging.info(f"Saving to disk: {os.path.join(FLAGS.output_path, filename)}")
    jnp.savez(os.path.join(FLAGS.output_path, filename), mean_auc_st=mean_auc_st, std_auc_st=std_auc_st, mean_auc_rst=mean_auc_rst, std_auc_rst=std_auc_rst) 

if __name__ == "__main__":
    
    app.run(main)