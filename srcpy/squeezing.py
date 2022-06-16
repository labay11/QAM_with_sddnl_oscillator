from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from joblib import Parallel, delayed

from model import build_system
from utils import DATA_PATH, PLOT_PATH, latexify


def local_data_path(nl_eta, nl_dis):
    path = DATA_PATH / Path(__file__).stem / 'a' / f'{nl_eta}_{nl_dis}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_plot_path(nl_eta, nl_dis):
    path = PLOT_PATH / Path(__file__).stem / 'a' / f'{nl_eta}_{nl_dis}'
    path.mkdir(parents=True, exist_ok=True)
    return path


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

    S = np.sqrt(EV[:, :, 1] - EV[:, :, 0]**2)
    cb = ax.pcolormesh(g2s, etas, S, cmap='magma', shading='gouraud', rasterized=True)
    ax.set(xlabel=r'$\gamma_m$', ylabel=r'$\eta_n$')
    cbar = fig.colorbar(cb)
    cbar.ax.set_ylabel('Squeezing')
    fig.savefig(outpath)


def plot_lines(files, outpath, labels=None):
    latexify(plt, type='paper', fract=0.25)

    if labels is None:
        labels = [None] * len(files)

    fig, ax = plt.subplots()
    for file, lbl in zip(files, labels):
        ev = np.real(np.load(file))
        betas, g, D = parse_filename(file.stem)
        ax.plot(betas, np.sqrt(ev[:, 1] - ev[:, 0]), label=lbl)

    ax.set(xlabel=r'$\eta_n/\gamma_m$', ylabel='Squeezing')
    ax.legend()

    fig.savefig(outpath)


def plot_all():
    dirpath = local_data_path(3, 2).parent

    for folder in dirpath.iterdir():
        n, m = map(int, folder.name.split('_'))
        savepath = local_plot_path(n, m)
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
    dirpath = local_data_path(n, m)
    files = [
        f
        for f in dirpath.iterdir()
        if f.suffix == '.npy' and not f.is_dir() and f.name.startswith('beta') and filter_by(parse_filename(f.stem))
    ]

    plot_lines(files, outpath, [label(parse_filename(f.stem)) for f in files])


if __name__ == '__main__':
    plot_comparison(
        4, 3,
        lambda x: x[2] == 1.5,
        local_plot_path(4, 3) / 'compare_d-1.5.pdf',
        lambda x: x[1])
