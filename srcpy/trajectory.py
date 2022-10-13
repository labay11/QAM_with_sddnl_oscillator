import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qutip import rand_ket, coherent, expect, mcsolve, mesolve, basis
from qutip.solver import Options
from qutip.fileio import qsave, qload
# from joblib import Parallel, delayed

from model import build_system
from utils import local_data_path, local_plot_path, amplitude, build_filename
from wigner import wigner_ss
from constants import POINTS


def _rand_ket(m, a=None, amp_max=None):
    if amp_max is None:
        return rand_ket(m)

    j = 0
    while j < 100:
        r = rand_ket(m)
        if np.abs(expect(a, r)) <= amp_max:
            return r
        j += 1

    raise RuntimeError('No state could be found with matching conditions.')


def evolve_mc(times, phi0, g1, g2, eta, D, n, m, dim):
    H, J = build_system(g1, g2, eta, D, n, m, dim, full_lv=False)

    opts = Options(num_cpus=4,
                   store_states=True,
                   store_final_state=True)
    data = mcsolve(H, phi0, times, c_ops=J, ntraj=1, options=opts)

    fname = local_data_path(__file__, n, m) / f'mc&{time.time_ns()}_{build_filename(**locals())}'
    qsave(data, str(fname))

    return data, fname


def evolve_me(ts, phi0, g1, g2, eta, D, n, m, dim):
    H, J = build_system(g1, g2, eta, D, n, m, dim, full_lv=False)

    opts = Options(num_cpus=4,
                   nsteps=5000,
                   store_states=True,
                   store_final_state=True)
    data = mesolve(H, phi0, ts, c_ops=J, options=opts)

    fname = local_data_path(__file__, n, m) / f'me&{time.time_ns()}_{build_filename(**locals())}'
    qsave(data, str(fname))

    return data, fname


def animate_trajectory(fpath, alpha_max=7.5):
    fname = fpath.name
    data = qload(str(fpath))

    states = np.mean(data.states, axis=0) if fname.startswith('mc') else data.states
    times = data.times
    N = len(times)

    fig, ax = plt.subplots(figsize=(6, 5))

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    xvec = np.linspace(-alpha_max, alpha_max, 200)

    def _calc_wigner(i):
        return times[i], wigner_ss(states[i], xvec)

    T, Rss = [], []
    nsmall = int(0.1 * N)
    for i in range(nsmall):
        T.append(times[i])
        Rss.append(np.array(wigner_ss(states[i], xvec)))
    for i in range(nsmall, N, 10):
        T.append(times[i])
        Rss.append(np.array(wigner_ss(states[i], xvec)))

    # _res = Parallel(n_jobs=5, verbose=1)(delayed(_calc_wigner)(i) for i in range(0, N, 10))
    # _t, _x = zip(*_res)
    Rss = np.array(Rss)
    # T = np.array(_t)
    N = len(T)
    norm = Normalize(Rss.min(), Rss.max())

    cf = ax.pcolormesh(xvec, xvec, Rss[0],
                       norm=norm, cmap='magma', shading='nearest', rasterized=True)
    fig.colorbar(cf, cax=cax)
    ax.set_xlabel(r'$\rm{Re}(\alpha)$')
    ax.set_ylabel(r'$\rm{Im}(\alpha)$')
    ax.set_title(r'$t = 0$')

    def animate(i):
        ax.clear()
        ax.pcolormesh(xvec, xvec, Rss[i],
                      norm=norm, cmap='magma', shading='nearest', rasterized=True)
        ax.set_title(rf'$t = {T[i]}$')

        return []

    interval = 0.1  # in seconds
    ani = animation.FuncAnimation(fig, animate, N, interval=interval * 1e3, blit=True)

    n, m = map(int, fpath.parent.name.split('_'))
    ani.save(local_plot_path(__file__, n, m) / (fname[:-3] + '.mp4'))

    return ani


if __name__ == '__main__':
    n = 4
    params = {
        'g1': 1.,
        'g2': POINTS[n][1][0],
        'eta': POINTS[n][1][1],
        'D': 0.4,
        'n': n,
        'm': n,
        'dim': POINTS[n][1][2]
    }

    beta = amplitude(**params)

    theta_mid = np.exp(1j * 2 * np.pi / n)
    phi0 = coherent(params['dim'], 2 * beta * np.exp(1j * np.pi / 8))

    times = np.logspace(-2, 2.2, 50000)

    data, fpath = evolve_me(times, phi0, **params)
    animate_trajectory(fpath)
