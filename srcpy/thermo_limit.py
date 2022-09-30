import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import cool

from utils import local_data_path, local_plot_path, parse_filename, latexify, cmap_clamp, build_filename


def read_files(n, m):
    dirpath = local_data_path(__file__, n, m)
    files = [
        (file, parse_filename(file.name[:-4].replace('-', '&')))
        for file in dirpath.iterdir()
        if file.name.endswith('.npy')
    ]

    return files


def filter_files(files, D, etas, dim, g1=1.):
    filtered = []

    for file, params in files:
        if params['D'] != D:
            continue

        if params['g1'] != g1:
            continue

        if params['dim'] != dim:
            continue

        x = params['etas']
        if len(x) != len(etas):
            continue

        if x[0] != etas[0] or x[-1] != etas[-1]:
            continue

        filtered.append((file, params))

    return sorted(filtered, key=lambda p: -p[1]['g2'])


def plot_thermodynamic_limit(etas, D, n, m, dim, g1=1.):
    files = filter_files(read_files(n, m), D, etas, dim, g1=g1)

    latexify(plt, type='paper', fract=0.4)

    Ns = [p['g1'] / p['g2'] for _, p in files]

    Nmin = 0
    Nmax = max(Ns)

    fig, axs = plt.subplots(nrows=2, sharex=True)

    # res = root(mf_amplitude, [mu, 1 * np.pi / n], args=(g1, g2, eta, D, n, m))

    for (file, params), N in zip(files, Ns):
        data = np.load(file)[:75]

        x = params['etas'][:75]

        y = np.abs(data[:, 0])
        y[y > np.mean(y) + 5 * np.std(y)] = np.nan

        axs[0].plot(x, y, c=cmap_clamp(cool, N, Nmin, Nmax), label=f'{int(N)}')

        y = np.imag(data[:, 2])
        y[y > np.mean(y) + 5 * np.std(y)] = np.nan

        axs[1].plot(x, y, c=cmap_clamp(cool, N, Nmin, Nmax), label=f'{int(N)}')

    axs[0].set(ylabel=r'$\langle \hat{n} \rangle / N$')
    axs[1].set(ylabel=r'Im$ \langle \hat{a}^n \rangle / N$', xlabel=r'$\eta/\gamma_1$')
    axs[0].legend()

    fig.savefig(local_plot_path(__file__, n, m) / build_filename(ext='.pdf', **locals()))


plot_thermodynamic_limit(np.linspace(0, 0.5, 100), 0.4, 3, 3, 200)
