from argparse import ArgumentParser

import matplotlib.pyplot as plt
from qutip import steadystate

from model import build_system
from wigner import plot_wigner
from utils import local_plot_path, latexify, amplitude, build_filename

latexify(plt, type='paper', fract=0.3)


def save_wigner_ss(**params):
    if 'g1' not in params:
        params['g1'] = 1.

    beta = amplitude(**params)
    if 'dim' not in params:
        params['dim'] = min(max(50, round(beta**2)), 200)

    L = build_system(**params)
    rho = steadystate(L)

    fig, ax = plt.subplots()

    plot_wigner(rho, fig=fig, ax=ax, alpha_max=round(2 * beta + 1), colorbar=True)

    fpath = local_plot_path(__file__, params['n'], params['m']) / (build_filename(**params) + '.pdf')
    fig.savefig(fpath)


if __name__ == '__main__':
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('-n', type=int, help='nl_eta')
    parser.add_argument('-m', type=int, help='nl_dis')
    parser.add_argument('--D', default=0.4, type=float, help='delta')
    parser.add_argument('--g1', default=1., type=float, help='gamma 1')
    parser.add_argument('--g2', type=float, default=0.5)
    parser.add_argument('--eta', type=float, default=2)
    parser.add_argument('--dim', type=int, default=50)

    args = parser.parse_args()

    save_wigner_ss(**vars(args))
