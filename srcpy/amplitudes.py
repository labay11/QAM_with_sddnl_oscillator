from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from model import build_system
from utils import amplitude, latexify, local_plot_path, local_data_path, parse_filename


LS = {3: '-', 4: '--', 5: ':'}


def squeezing_amplitude(etas, n, m, g=0.4, d=0.4):
    N = len(etas)

    EV = np.zeros((N, 2))

    num = np.diag(range(200))

    for j in range(1, N):
        beta = amplitude(g, etas[j] * g, n, m)
        dim = min(max(int(round(beta * 3)), 80), 200)
        L = build_system(1.0, g, etas[j] * g, d, n, m, dim)
        # ss = steadystate.iterative(H, J)
        try:
            ss = qt.steadystate(L).data.toarray()
            EV[j, 0] = beta
            EV[j, 1] = np.trace(num[:dim, :dim] @ ss)
        except Exception:
            EV[j, :] = np.nan

    np.save(local_data_path(__file__, n, m) / f"eta-{etas[0]}-{etas[-1]}-{N}_d-{d}_g-{g}.npy", EV)


def comparison(exponents, g2=0.4):
    latexify(plt, type='paper', fract=0.25, palette='default_mk')

    fig, ax = plt.subplots()
    ax.set(xlabel=r'$\eta_n / \gamma_m$', ylabel='Mean field amplitude')

    x = np.linspace(0, 2, 500)

    for nl_eta, nl_dis in exponents:
        y = amplitude(g2, x * g2, nl_eta, nl_dis)
        ax.plot(x, y, label=f'${nl_eta} - {nl_dis}$', markevery=249)

    ax.legend(title=r'$n - m$', ncol=2)
    fig.savefig(local_plot_path(__file__) / 'amplitude_comparison.pdf')


def _clean_up(x):
    dists = np.abs(x - np.roll(x, 1))**2
    y = x.copy()
    y = (np.roll(x, -2) + np.roll(x, 2)) * 0.5
    y[:2] = x[:2]
    y[-2:] = x[-2:]
    return y


def plot_mf_n(g=0.4, d=0.4):
    dirpath = local_data_path(__file__)

    mnms = sorted(
        [tuple(map(int, folder.name.split('_'))) for folder in dirpath.iterdir()],
        key=lambda x: (x[0], x[1])
    )

    latexify(plt, type='paper', fract=0.25, palette='default')

    endname = f'd-{d}_g-{g}.npy'

    fig, ax = plt.subplots()
    ax.set(xlabel=r'$\eta_n / \gamma_m$', ylabel=r'$|\beta - R|^2$', yscale='log')
    etas = np.linspace(0, 20, 500)

    for n, m in mnms:
        if n == 5:
            continue
        nmdir = dirpath / f'{n}_{m}'
        f = next(f for f in nmdir.iterdir() if f.name.endswith(endname))

        data = np.load(f)
        data[data < 0] = np.nan
        ax.plot(etas, np.abs(np.sqrt(data[:, 1]) - amplitude(g, etas * g, n, m))**2, label=f'${n} - {m}$', ls=LS[n])

    ax.legend(title=r'$n - m$', ncol=2)
    fig.savefig(local_plot_path(__file__) / 'mf_quantum_amplitude.pdf')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-n', type=int, help='nl_eta == nl_dis')
    parser.add_argument('-m', type=int, help='method to use (1: me, 2: deterministic)')
    parser.add_argument('--bmin', type=float, help='beta min')
    parser.add_argument('--bmax', type=float, help='beta max')
    parser.add_argument('--bnum', type=int, help='beta num')

    parser.add_argument('-p', '--plot', action='store_true')

    args = parser.parse_args()

    if args.plot:
        # comparison([(2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (4, 5)])
        plot_mf_n()
    else:
        etas = np.linspace(args.bmin, args.bmax, args.bnum)
        squeezing_amplitude(etas, args.n, args.m)
