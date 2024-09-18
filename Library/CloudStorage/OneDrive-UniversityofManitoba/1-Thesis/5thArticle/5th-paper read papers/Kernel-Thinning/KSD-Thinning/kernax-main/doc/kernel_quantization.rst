Maximum mean disrepancy quantization
====================================

This notebook illustrates how to use :class:`KernelQuantization` for compressing a sample [1,2]. Let's generate a Gaussian toy sample.

.. code::

    import numpy as np
    x = np.random.randn(1000, 2)

Now that we have a toy sample, let's pick a kernel function. Here, we use the energy kernel function. In this case, the maximum mean discrepancy reduces to the energy distance [3].

.. code::

    from kernax.kernels import Energy
    from kernax import KernelQuantization

    quant_fn = KernelQuantization(X, kernel_fn=Energy)
    idx = quant_fn(m = 1_000)

The output `idx` gathers the selected indices. Let's plot the result.

.. code::

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X[:, 0], X[:, 1], ls="", marker="o", color="k", label="Initial sample")
    ax.plot(X[idx, 0], X[idx, 1], ls="", marker="o", color="r", label="Selected sample")
    ax.legend(fontsize=14)

Note that any custom kernel function can be used. An example is shown below.

.. code::

    import jax.numpy as jnp
    from jax import Array

    def custom_kernel(x: Array, y: Array) -> float:
        # implement your own kernel
        # example: linear kernel
        kxy = jnp.dot(x, y)
        return kxy

References
~~~~~~~~~~

[1] Chen, Y., Welling, M., & Smola, A. (2012).
Super-samples from kernel herding. arXiv preprint arXiv:1203.3472

[2] Teymur, O., Gorham, J., Riabiz, M., & Oates, C. (2021, March).
Optimal quantisation of probability measures using maximum mean discrepancy. In International Conference on Artificial Intelligence and Statistics (pp. 1027-1035). PMLR.

[3] Sejdinovic, D., Sriperumbudur, B., Gretton, A., & Fukumizu, K. (2013).
Equivalence of distance-based and RKHS-based statistics in hypothesis testing. The annals of statistics, 2263-2291.