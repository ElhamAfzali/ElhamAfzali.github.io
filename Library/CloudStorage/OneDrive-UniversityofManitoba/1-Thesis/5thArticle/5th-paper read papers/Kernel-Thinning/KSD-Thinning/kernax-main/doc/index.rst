.. BaTorch documentation master file, created by
   sphinx-quickstart on Thu Dec  9 09:28:21 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Kernax
=================

Kernax is a small package that implements KSD-based algorithms for post-processing MCMC outputs. It is based on JAX and works on CPU as well as GPU.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started
   stein_thinning
   kernel_quantization

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   thinning_algorithms
   quantization_algorithms
   kernels
   discrepancies
   samplers
   utils