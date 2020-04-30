import numpy as np


def log_multivariate_gaussian(x, mu, V, Vinv):
    """Evaluate the log of a multivariate Gaussian"""
    
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)
    Vinv = np.asarray(Vinv, dtype=float)

    ndim = x.shape[-1]
    x_mu = x - mu
    
    Vshape = V.shape
    V = V.reshape([-1, ndim, ndim])

    logdet = np.log(np.array([np.linalg.det(V[i])
                              for i in range(V.shape[0])]))
    logdet = logdet.reshape(Vshape[:-2])

    xVI = np.sum(x_mu.reshape(x_mu.shape + (1,)) * Vinv, -2)
    xVIx = np.sum(xVI * x_mu, -1)

    return -0.5 * ndim * np.log(2 * np.pi) - 0.5 * (logdet + xVIx)
