# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import jax
import jax.numpy as jnp

import blackjax
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.mala import MALAState
from kernax.types import Array, PRNGKey

from typing import Callable

def inference_loop(rng_key, step_fn, initial_state, num_samples):
    """Inference loop with lax.scan"""
    @jax.jit
    def one_step(state, rng_key):
        state, info = step_fn(rng_key, state)
        return state, (info, state)
    keys = jax.random.split(rng_key, num_samples)
    _, (infos, states) = jax.lax.scan(one_step, initial_state, keys)
    return infos, states

def rmh(logprob_fn: Callable,
        init_positions: Array,
        num_samples: int,
        sigma: Array,
        rng_key: PRNGKey) -> HMCState:
    """Wrapper of the HMC algorithm implemented in BlackJAX.

    Parameters
    ----------

    logprob_fn
        Callable function that returns the log-probability of the target
    init_positions
        Initial guess for the HMC algorithm
    num_samples
        Number of iterations (burn-in included)
    sigma
        Scaling of the Gaussian proposal (vector)
    rng_key
        A JAX PRNGKey.

    Returns
    -------

    A tuple of states and informations
    """

    kern = blackjax.rmh(logprob_fn, sigma)
    step_fn = jax.jit(kern.step)
    init_state = kern.init(init_positions)
    infos, states = inference_loop(rng_key, step_fn, init_state, num_samples)
    return infos, states

def hmc(logprob_fn: Callable,
        init_positions: Array,
        num_samples: int,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
        rng_key: PRNGKey) -> HMCState:
    """Wrapper of the HMC algorithm implemented in BlackJAX.

    Parameters
    ----------

    logprob_fn
        Callable function that returns the log-probability of the target
    init_positions
        Initial guess for the HMC algorithm
    num_samples
        Number of iterations (burn-in included)
    inverse_mass_matrix
        Flattened inverse mass matrix
    num_integration_steps
        Number of integration steps
    rng_key
        A JAX PRNGKey.

    Returns
    -------

    A tuple of states and informations
    """

    kern = blackjax.hmc(logprob_fn, step_size, inverse_mass_matrix, num_integration_steps)
    step_fn = jax.jit(kern.step)
    init_state = kern.init(init_positions)
    infos, states = inference_loop(rng_key, step_fn, init_state, num_samples)
    return infos, states

def nuts(logprob_fn: Callable,
         init_positions: Array,
         num_samples: int,
         step_size: float,
         inverse_mass_matrix: Array,
         rng_key: PRNGKey) -> HMCState:
    """Wrapper of the NUTS algorithm implemented in BlackJAX.

    Parameters
    ----------

    logprob_fn
        Callable function that returns the log-probability of the target
    init_positions
        Initial guess for the HMC algorithm
    num_samples
        Number of iterations (burn-in included)
    step_size
        Step size of the leapfrog integrator
    inverse_mass_matrix
        Flattened inverse mass matrix
    rng_key
        A JAX PRNGKey.

    Returns
    -------

    A tuple of states and informations
    """

    kern = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)
    step_fn = jax.jit(kern.step)
    init_state = kern.init(init_positions)
    infos, states = inference_loop(rng_key, step_fn, init_state, num_samples)
    return infos, states

def mala(logprob_fn: Callable,
         init_positions: Array,
         num_samples: int,
         step_size: float,
         rng_key: PRNGKey) -> MALAState:
    """Wrapper of the MALA algorithm implemented in BlackJAX.

    Parameters
    ----------

    logprob_fn
        Callable function that returns the log-probability of the target
    init_positions
        Initial guess for MALA
    num_samples
        Number of iterations (burn-in included)
    step_size
        Step size of the integrator
    rng_key
        A JAX PRNGKey.

    Returns
    -------

    A tuple of states and informations
    """

    kern = blackjax.mala(logprob_fn, step_size)
    step_fn = jax.jit(kern.step)
    init_state = kern.init(init_positions)
    infos, states = inference_loop(rng_key, step_fn, init_state, num_samples)
    return infos, states