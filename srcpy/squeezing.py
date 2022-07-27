import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from model import build_system
from utils import local_data_path, local_plot_path, latexify


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


def _expect(*params, ops):
    try:
        L = build_system(*params)
        ss = qt.steadystate(L)
        return [np.real(qt.expect(op, ss)) for op in ops]
    except Exception:
        return [np.nan] * len(ops)


def plot_map(file, outpath):
    latexify(plt, type='paper', fract=0.25)

    fig, ax = plt.subplots()

    etas, g2s, D = parse_filename(file.name)

    EV = np.real(np.load(file))

    S = np.sqrt(EV[:, :, 1] - EV[:, :, 0]**2) / EV[:, :, 0]
    cb = ax.pcolormesh(g2s, etas, S, cmap='magma', shading='gouraud', rasterized=True)
    ax.set(xlabel=r'$\gamma_m$', ylabel=r'$\eta_n$')
    cbar = fig.colorbar(cb)
    cbar.ax.set_ylabel('Squeezing')
    fig.savefig(outpath)


def plot_lines(files, outpath, labels=None, right_amp=False):
    latexify(plt, type='paper', fract=0.25)

    if labels is None:
        labels = [None] * len(files)

    fig, ax = plt.subplots()

    for file, lbl in zip(files, labels):
        ev = np.real(np.load(file))
        betas, g, D = parse_filename(file.stem)
        sq = np.sqrt(ev[:, 1] - ev[:, 0]**2)
        sq[sq < 0] = np.nan
        ax.plot(betas[10:], sq[10:], label=lbl)

    ax.set(xlabel=r'$\beta$', ylabel='Squeezing', yscale='log')
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
            elif file.name.startswith('g'):
                # g-eta plot
                plot_map(file, savepath / f'{file.stem}.pdf')
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
    ds = [0.1, 0.2, 0.5, 1, 1.5]
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


if __name__ == '__main__':
    plot_nm_comparisons()

    # plot_all()
