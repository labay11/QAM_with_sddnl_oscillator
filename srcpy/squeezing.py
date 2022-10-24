from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from model import build_system
from ems import metastable_states
from utils import local_data_path, local_plot_path, latexify, amplitude


def parse_linspace(data):
    parts = data.split('-')
    return np.linspace(float(parts[1]), float(parts[2]), int(parts[3]))


def parse_filename(fname):
    parts = fname.split('_')

    if 'beta' in parts[0]:
        betas = parse_linspace(parts[0])
        D = float(parts[1].split('-')[1])
        g = float(parts[2].split('-')[1])
        return betas, g, D
    else:
        gs = parse_linspace(parts[0])
        etas = parse_linspace(parts[1])
        D = float(parts[2].split('-')[1])
        return etas, gs, D


def squeezing_amplitude(betas, d, n, m, g=0.2):
    N = len(betas)

    EV = np.zeros((N, 2, n))

    num = np.diag(range(80))
    num2 = num**2

    for j in range(1, N):
        beta = amplitude(g, betas[j] * g, n, m)
        dim = min(max(int(round(beta * 3)), 40), 80)
        try:
            ems, *_ = metastable_states(g, betas[j] * g, d, n, m, dim)
            EV[j, 0, :] = [np.trace(num[:dim, :dim] @ ss) for ss in ems[1:]]
            EV[j, 1, :] = [np.trace(num2[:dim, :dim] @ ss) for ss in ems[1:]]
        except Exception:
            EV[j, :, :] = np.nan

    np.save(local_data_path(__file__, n, m) / f"beta-{betas[0]}-{betas[-1]}-{N}_d-{d}_g-{g}.npy", EV)


def plot_lines(files, outpath, labels=None, right_amp=False):
    if not files:
        return

    latexify(plt, type='preprint', fract=(0.33, 0.25))

    if labels is None:
        labels = [None] * len(files)

    fig, ax = plt.subplots()

    for file, lbl in zip(files, labels):
        ev = np.real(np.load(file))
        betas, g, D = parse_filename(file.stem)
        sq = np.sqrt(ev[:, 1] - ev[:, 0]**2) / np.sqrt(ev[:, 0])
        sq[sq < 0] = np.nan
        ax.plot(betas[10:], sq[10:], label=lbl)

    ax.plot(betas[10:], np.ones(len(betas) - 10), c='k', ls=':', label=r'$1$')

    ax.set(xlabel=r'$\eta_n/\gamma_m$', ylabel='Squeezing')
    ax.legend(ncol=2)

    fig.savefig(outpath)


def plot_all():
    dirpath = local_data_path(__file__)

    for folder in dirpath.iterdir():
        n, m = map(int, folder.name.split('_'))
        savepath = local_plot_path(__file__, n, m)
        for file in folder.iterdir():
            if file.suffix != '.npy' or file.is_dir():
                continue

            if file.name.startswith('beta'):
                # single line plot
                plot_lines([file], savepath / f'{file.stem}.pdf')
            else:
                continue


def plot_comparison(n, m, filter_by, outpath, label):
    dirpath = local_data_path(__file__, n, m)
    files = [
        f
        for f in dirpath.iterdir()
        if f.suffix == '.npy' and not f.is_dir() and f.name.startswith('beta') and filter_by(parse_filename(f.stem))
    ]

    plot_lines(files, outpath, [label(parse_filename(f.stem)) for f in files])


def plot_gd_comparisons():
    ds = [0.1, 0.2, 0.5, 1.5]
    gs = [0.1, 0.2, 0.4, 0.8, 1.2]

    dirpath = local_data_path(__file__)

    for folder in dirpath.iterdir():
        n, m = map(int, folder.name.split('_'))

        for d in ds:
            plot_comparison(
                n, m,
                lambda x: x[2] == d,
                local_plot_path(__file__, n, m) / f'compare_d-{d}.pdf',
                lambda x: x[1])

        for g in gs:
            plot_comparison(
                n, m,
                lambda x: x[1] == g,
                local_plot_path(__file__, n, m) / f'compare_g-{g}.pdf',
                lambda x: x[2])


def plot_nm_comparisons(d=None, g=None):
    ds = [0.1, 0.2, 0.5, 1, 1.5]
    gs = [0.1, 0.2, 0.4, 0.8, 1.2]

    dirpath = local_data_path(__file__)
    outpath = local_plot_path(__file__, 'n', 'm')

    mnms = sorted(
        [tuple(map(int, folder.name.split('_'))) for folder in dirpath.iterdir()],
        key=lambda x: (x[0], x[1])
    )
    labels = [f'${n} - {m}$' for n, m in mnms]

    for d in ds:
        for g in gs:
            fpaths = [dirpath / f'{n}_{m}' / f'beta-0.0-20.0-500_d-{d}_g-{g}.npy' for n, m in mnms]
            fpaths = [f for f in fpaths if f.exists()]
            plot_lines(fpaths, outpath / f'd-{d}_g-{g}.pdf', labels)
            for n_ in [3, 4, 5]:
                fpaths = [dirpath / f'{n}_{m}' / f'beta-0.0-20.0-500_d-{d}_g-{g}.npy' for n, m in mnms if n == n_]
                fpaths = [f for f in fpaths if f.exists()]
                plot_lines(fpaths, outpath / f'n-{n_}_d-{d}_g-{g}.pdf', [f'${n} - {m}$' for n, m in mnms if n == n_])


def plot_gd_comparison(g, d):
    nn = [3, 4, 5]

    dirpath = local_data_path(__file__)
    outpath = local_plot_path(__file__, 'n', 'm')

    mnms = sorted(
        [tuple(map(int, folder.name.split('_'))) for folder in dirpath.iterdir()],
        key=lambda x: (x[0], x[1])
    )

    fig, ax = plt.subplots(ncols=3)
    return


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-g', type=float, help='g2', default=0.2)
    parser.add_argument('-d', type=float, help='delta', default=0.4)
    parser.add_argument('-n', type=int, help='nl_eta == nl_dis')
    parser.add_argument('-m', type=int, help='method to use (1: me, 2: deterministic)')
    parser.add_argument('--bmin', type=float, help='beta min')
    parser.add_argument('--bmax', type=float, help='beta max')
    parser.add_argument('--bnum', type=int, help='beta num')

    parser.add_argument('-p', '--plot', action='store_true')

    args = parser.parse_args()

    if args.plot:
        plot_nm_comparisons()
    else:
        betas = np.linspace(args.bmin, args.bmax, args.bnum)
        squeezing_amplitude(betas, args.d, args.n, args.m, g=args.g)
