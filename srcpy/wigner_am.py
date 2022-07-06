import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

TWO_PI = 2 / np.pi


def lobes(alpha, beta):
    return np.exp(-2 * np.abs(alpha - beta)) * TWO_PI


def kernel(alpha1, alpha2, beta, n):
    return sum(
        lobes(alpha1, beta * np.exp(1j * (2 * j + 1) * np.pi / n)) *
        lobes(alpha1, beta * np.exp(1j * (2 * j + 1) * np.pi / n))
        for j in range(n)
    )


def ri_to_rphase(xvec):
    """converts a wigner from re/im space to amplitude/phase space"""
    x, y = np.meshgrid(xvec, xvec)
    alpha = x + 1j * y
    R, phi = np.abs(alpha), np.angle(alpha) + np.pi
    return R, phi, alpha


def evolve(rho0, beta, n, t_max=100, h=1e-2, alpha_max=None, div=100):
    if alpha_max is None or alpha_max <= 0:
        alpha_max = 2 * beta

    xvec = np.linspace(-alpha_max, alpha_max, div)
    W0 = qt.wigner(rho0, xvec, xvec, method='clenshaw')

    W = W0.copy()
    R, phi, alpha = ri_to_rphase(xvec)
    dx = 2 * alpha_max / div

    t = 0
    yield W0

    while t < t_max:
        t += h

        W += (1 - h) * W

        for m in range(div):
            for n in range(div):
                W *= kernel(alpha, alpha[m, n], beta, n) * W[m, n] * R[m, n] * dx**2

        yield W

    return W


rho0 = qt.coherent_dm(50, 2 * np.exp(1j * 2 * np.pi / 9))
xvec = np.linspace(-2 * 3, 2 * 3, 100)

for w in evolve(rho0, 3., 3):
    cf = plt.pcolormesh(xvec, xvec, w, cmap='magma', shading='nearest', rasterized=True)
    plt.colorbar(cf)
    plt.show()
