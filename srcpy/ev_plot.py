import numpy as np
import matplotlib.pyplot as plt

from utils import TOL, local_data_path, local_plot_path, parse_filename, latexify


def join_dir(dirpath):
    files = [
        (int(fn.stem), fn)
        for fn in dirpath.iterdir()
    ]
    files = sorted(files, key=lambda x: x[0])

    params = parse_filename(dirpath.name)
    Ng, Ne = len(params['g2s']), len(params['etas'])
    eigvals = np.load(files[0][1]).shape[-1]

    EV = np.zeros((Ng, Ne, eigvals), dtype=complex)
    for idx, path in files:
        # if use_rows:
        #     EV[idx - 1, :, :] = np.load(path)
        # else:
        EV[:, idx, :] = np.load(path)

    filepath = dirpath.parent / (dirpath.name + '.npy')
    np.save(filepath, EV)


def join_all_dirs():
    data_path = local_data_path('ev.py')

    for nm_path in data_path.iterdir():
        for fpath in nm_path.iterdir():
            fname = fpath.with_suffix('.npy')
            if fname.exists():
                continue
            if fpath.is_dir():
                try:
                    join_dir(fpath)
                except:
                    print(fpath)


def plot_gap(filepath, fig=None, ax=None, colorbar=True, levels=[0.01, 0.1, 0.5], colors=['C1', 'C2', 'C0']):
    EV = np.load(filepath)

    Evr = np.abs(np.real(EV))
    Evr[Evr < TOL] = np.nan
    Evi = np.abs(np.imag(EV))
    Evi[Evi < TOL] = np.nan

    params = parse_filename(filepath.name[:-4])
    n, m = list(map(int, filepath.parent.name.split('_')))

    gap = np.log10(Evr[:, :, n - 1] / Evr[:, :, n]).T

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    g2s, etas = params['g2s'], params['etas']

    cb = ax.pcolormesh(g2s, etas, gap, cmap='magma', shading='gouraud', rasterized=True)
    if len(levels) > 0:
        CS = ax.contour(g2s, etas, Evr[:, :, n - 1].T, levels, colors=colors or [f'C{c}' for c in range(len(levels))])
        # for col in CS.collections:
        #     if col.get_paths():
        #         v = col.get_paths()[0].vertices
        #         x, y = v[:, 0], v[:, 1]
        #         res = linregress(x, y)
        #         print(res.slope, res.intercept)

    if colorbar:
        ticks = list(range(int(np.nanmin(gap)), 1))
        if len(ticks) > 10:
            ticks = ticks[::2]
        cbar = plt.colorbar(cb, ax=ax, ticks=ticks)
        cbar.ax.set_ylabel(r'$\log_{10}' + rf'\tau_{n + 1}/\tau_{n}$')

    ax.set(xlabel=rf'$\gamma_{n}/\gamma_1$', ylabel=rf'$\eta/\gamma_1$')

    fig.savefig(local_plot_path('ev.py', n, m) / (filepath.name[:-4] + '.pdf'))

    return fig, ax, CS


def plot_all():
    latexify(plt, type='paper', fract=0.3)

    data_path = local_data_path('ev.py')

    for nm_path in data_path.iterdir():
        for fpath in nm_path.iterdir():
            if fpath.is_file() and fpath.suffix == '.npy':
                try:
                    plot_gap(fpath)
                except:
                    print(fpath)


if __name__ == '__main__':
    join_all_dirs()
    plot_all()
