import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from joblib import Parallel, delayed

from model import classicallity
from ems import metastable_states
from utils import local_data_path, local_plot_path, driving_dissipation_ratio, latexify


def _internal(beta, g2, d, nl_eta, nl_dis):
    eta = driving_dissipation_ratio(beta, nl_eta, nl_dis) * g2
    dim = 120
    ems, ls, rs, ps = metastable_states(g2, eta, d, nl_eta, nl_dis, dim=dim)
    _, c, _ = classicallity(ls, ems)
    return c


def compute_classicality(betas, nl_eta, nl_dis, g2=0.4, d=0.4, parallel=True):
    if parallel:
        res = Parallel(n_jobs=4)(delayed(_internal)(beta, g2, d, nl_eta, nl_dis) for beta in betas)
    else:
        res = [_internal(beta, g2, d, nl_eta, nl_dis) for beta in betas]

    return np.array(res)


def join_files():
    import shutil
    parentdir = local_data_path(__file__)

    for nmdir in parentdir.iterdir():
        for g2dir in nmdir.iterdir():
            if not g2dir.is_dir():
                continue
            files = sorted([f for f in g2dir.iterdir()], key=lambda f: int(f.stem))
            data = [np.load(f) for f in files]
            np.save(nmdir / f'{g2dir.name}.npy', np.array(data))
            if len(files) == 51:
                shutil.rmtree(g2dir)


def plot_comparison():
    latexify(plt, type='beamer43', fract=(0.8, 0.6), palette='default')

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set(xlabel=r'$\beta$', ylabel='Classicality')

    parentdir = local_data_path(__file__)
    c = 0
    for nmdir in parentdir.iterdir():
        n, m = list(map(int, nmdir.name.split('_')))
        g2s = [f for f in nmdir.iterdir() if f.name.startswith('g') and f.name.endswith('.npy')]
        g2s = sorted(g2s, key=lambda x: float(x.name[:-4].split('_')[-1]))

        data = np.load(g2s[0])
        for j in range(len(g2s) - 1):
            data += np.load(g2s[j + 1])
        data /= len(g2s)
        data[data > 1e8] = np.nan

        ax.plot(data[:, 0], data[:, 1], c=f'C{c}', label=rf'${n} - {m}$')

        c += 1

    ax.legend()
    fig.savefig(local_plot_path(__file__) / 'comparison.pdf')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-g', type=float, help='g2')
    parser.add_argument('-j', type=int, help='index')
    parser.add_argument('-nl', type=int, help='nl_eta == nl_dis')
    parser.add_argument('-ml', type=int, help='method to use (1: me, 2: deterministic)')
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('--join', action='store_true')

    args = parser.parse_args()

    if args.join:
        join_files()
    elif args.plot:
        plot_comparison()
    else:
        fpath = local_data_path(__file__, args.nl, args.ml) / f'g_{args.g}'
        fpath.mkdir(parents=True, exist_ok=True)
        beta = 0.01 + (6 - 0.01) * args.j / 50.
        C = _internal(beta, args.g, 0.4, args.nl, args.ml)
        print(beta, C)
        np.save(fpath / f'{args.j}.npy', [beta, C])
