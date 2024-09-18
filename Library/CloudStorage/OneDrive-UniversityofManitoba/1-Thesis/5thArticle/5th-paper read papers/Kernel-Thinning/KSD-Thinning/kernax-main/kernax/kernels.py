# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import jax
import jax.numpy as jnp
from typing import Callable
from kernax.types import Array


def IMQ(x: Array, y: Array, lengthscale: float):
    """Inverse multi-quadratric kernel function.

    Parameters
    ----------

    x
        Vector of dimension :math:`d`
    y
        Vector of dimension :math:`d`
    lengthscale
        Scalar lengthscale / bandwidth

    Returns
    -------

    Value of the kernel function k(x,y)
    """
    return 1.0/jnp.sqrt(1.0 + jnp.sum((x-y)**2)/lengthscale**2)


def Gaussian(x: Array, y: Array, lengthscale: float):
    """Gaussian kernel function.

    Parameters
    ----------

    x
        Vector of dimension :math:`d`
    y
        Vector of dimension :math:`d`
    lengthscale
        Scalar lengthscale / bandwidth

    Returns
    -------

    Value of the kernel function k(x,y)
    """
    return jnp.exp(-0.5*jnp.sum((x-y)**2)/lengthscale**2)


def Energy(x: Array, y: Array):
    """Distance induced kernel function.

    Parameters
    ----------

    x
        Vector of dimension :math:`d`
    y
        Vector of dimension :math:`d`

    Returns
    -------

    Value of the kernel function k(x,y)
    """
    # return jnp.sqrt(jnp.sum(x**2)) + jnp.sqrt(jnp.sum(y**2)) - jnp.sqrt(jnp.sum((x-y)**2))
    return jnp.sqrt(x@x) + jnp.sqrt(y@y) - jnp.sqrt(jnp.clip(x@x + y@y - 2*x@y, a_min=0))


def SteinIMQ(x: Array, sx: Array, y: Array, sy: Array, lengthscale: float):
    """Langevin Stein kernel with the IMQ as the underlying kernel.

    Parameters
    ----------

    x
        Vector of dimension :math:`d`
    sx
        Score functon evaluated at x, vector of dimension :math:`d`
    y
        Vector of dimension :math:`d`
    sy
        Score function evaluetaed at y, vector of dimension :math:`d`

    Returns
    -------

    Value the Stein IMQ kernel :math:`k_p(x,y)`
    """

    d = len(x)
    sqdist = jnp.sum((x-y)**2)
    qf = 1.0/(1.0 + sqdist/lengthscale**2)
    t3 = jnp.dot(sx, sy) * jnp.sqrt(qf)
    t2 = (1.0/lengthscale**2) * (d + jnp.dot(sx-sy,x-y)) * qf**(3/2)
    t1 = (-3.0/lengthscale**4)*sqdist * qf**(5/2)
    return t1 + t2 + t3


def SteinGaussian(x: Array, sx: Array, y: Array, sy: Array, lengthscale: float):
    """Langevin Stein kernel with the Gaussian kernel as the underlying kernel.

    Parameters
    ----------

    x
        Vector of dimension :math:`d`
    sx
        Score functon evaluated at x, vector of dimension :math:`d`
    y
        Vector of dimension :math:`d`
    sy
        Score function evaluetaed at y, vector of dimension :math:`d`

    Returns
    -------

    Value the Stein Gaussian kernel :math:`k_p(x,y)`
    """

    d = len(x)
    sqdist = jnp.sum((x-y)**2)
    kxy = jnp.exp(-0.5*sqdist/lengthscale**2)
    t1 = d/lengthscale**2 - sqdist/lengthscale**4
    t2 = jnp.dot(sx-sy, x-y)/lengthscale**2
    t3 = jnp.dot(sx, sy)
    return kxy*(t1 + t2 + t3)


def GetSteinFn(kernel_fn: Callable):
    """Helper that builds the Stein kernel function :math:`k_p(x,y)` given an arbitrary underlying kernel :math:`k(x,y)`.

    Parameters
    ----------

    kernel_fn
        Callable kernel function of the form :math:`(x,y) \mapsto k(x,y)`. Any hyparameters such as the lengthscale should be fixed with `jax.tree_util.Partial`.

    Returns
    -------

    Callable function of the form :math:`(x, s_x, y, s_y) \mapsto k_p(x, s_x, y, s_y)` where :math:`x` and :math:`y` are the vectors of observations and :math:`s_x` and :math:`s_y` are the associated scores.
    """

    grad1_fn = jax.grad(kernel_fn, argnums=0)
    grad2_fn = jax.grad(kernel_fn, argnums=1)
    hessian_fn = jax.jacobian(grad2_fn, argnums=0)

    def kp_fn(x, sx, y, sy):
        return jnp.dot(sx, sy)*kernel_fn(x, y) + jnp.dot(sx, grad2_fn(x, y)) + jnp.dot(sy, grad1_fn(x, y)) + jnp.trace(hessian_fn(x, y))
    return kp_fn
