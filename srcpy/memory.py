import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import latexify, local_plot_path

DATA_PATH = os.path.expanduser('~/Documents/data/gqvdp/memory_test')


def _method_avg(Ps, method, n):
    N, _, nl_eta = Ps.shape
    if method == 1:
        p = 0
        up = 0
        for j in range(nl_eta):
            nj = np.count_nonzero(n == j)
            p += np.mean(Ps[n == j, :, j], axis=0)
            up += np.real(np.std(Ps[n == j, :, j], axis=0))**2
            print(j, nj, Ps[n == j, :, j].shape)
        p /= nl_eta
        up = np.sqrt(up) / nl_eta

        # b = np.array([Ps[j, :, n[j]] for j in range(N)])
        # p = np.real(np.mean(b, axis=0))
    else:
        p = np.count_nonzero(n == n[:, np.newaxis], axis=1) / N
        up = None

    return np.real(p), up


def plot_memory_data(fig, ax, nl_eta, nl_dis, dirpath, files, adag=True, method=0, **ln_kw):
    data = []
    for name in files:
        if name.endswith('.npy'):
            try:
                data.append(np.real(np.load(os.path.join(dirpath, name))))
            except Exception as e:
                print(name, e)
                os.remove(os.path.join(dirpath, name))
    data = np.array(data)
    print(data.shape)

    Ps = data[:, :, 1:nl_eta + 1]
    print(Ps.shape)
    if data.shape[2] == 1 + 2 * nl_eta:
        overlaps = data[:, :, nl_eta + 1:]
        P2s = np.ones(Ps.shape, dtype=complex)
        for k in range(nl_eta - 1):
            P2s[:, :, k] *= overlaps[:, :, k]
            for j in range(k + 1, nl_eta):
                P2s[:, :, j] *= (1 - overlaps[:, :, k])
    else:
        P2s = None

    n = np.argmax(Ps, axis=2)
    p, up = _method_avg(Ps, method, n[:, 0])

    # times = filedata[4].split('-')
    # times = np.linspace(0, float(times[0]), int(times[1]))
    times = np.real(data[0, :, 0])
    ax.plot(times, p, **ln_kw)
    # if up is not None:
    #     ax.fill_between(times, y1=p-up, y2=p+up, alpha=0.5, **ln_kw)

    if P2s is not None:
        p2, up2 = _method_avg(P2s, method, n[:, 0])
        ln_kw['label'] = None
        ax.plot(times, p2, ls='--', **ln_kw)
        # if up is not None:
        #     ax.fill_between(times, y1=p2-up2, y2=p2+up2, alpha=0.5, **ln_kw)
    else:
        p2 = None

    return times, p, p2


def _parse_files(n, adag=True):
    dirpath = os.path.join(DATA_PATH, "adag" if adag else "a", f'{n}_{n}', 'julia')

    possible_files = list(set(['_'.join(f.split('_')[:2]) for f in os.listdir(dirpath)]))
    if len(possible_files) == 0:
        print('No file')
        return None
    elif len(possible_files) == 1:
        file = possible_files[0]
    else:
        print(f'Choose options for {n}:\n' + '\n'.join([f'{j}: {name}' for j, name in enumerate(possible_files)]))
        j = int(input('Which one? '))
        file = possible_files[j]

    return dirpath, file


def plot_decay_comparison(adag=False, coh=True, method=1):
    latexify(plt, 0.5, margin_x=0, palette='qualitative')

    fig, ax = plt.subplots()
    ax.grid(False)
    files = [_parse_files(n, adag) for n in [3, 4]]

    for (dirpath, dirname), n, l in zip(files, [3, 4], ['a', 'b']):
        g2, eta = list(map(float, dirname.split('_')))

        f = sorted([f for f in os.listdir(dirpath) if f.startswith(dirname)],
                   key=lambda x: -int(x.split('_')[-1]))[0]

        filetrajs = os.listdir(os.path.join(dirpath, f))
        print(f'{len(filetrajs)} trajs found', f)
        times, *_ = plot_memory_data(fig, ax, n, n, os.path.join(dirpath, f), filetrajs,
                                     adag=adag, method=method, c=f'C{n-3}', label=rf'$n = {n}$')
    ax.set(xlabel=r'$\gamma_1 t$', xscale='log', ylabel='Success probability')

    ax.legend()

    fig.tight_layout(pad=0.02)
    fig.savefig(local_plot_path(__file__) / f'memory_{"adag" if adag else "a"}_{method}_julia_comp.pdf')


def plot_time_only(adag=False, coh=True, method=0):
    latexify(plt, 0.25, paper=2, margin_x=0, palette='qualitative')

    fig, time_axs = plt.subplots(nrows=2, sharex=True)
    files = [_parse_files(n, adag) for n in [3, 4]]

    D = 0.4

    cmap = cm.get_cmap('Set1').reversed()

    for (dirpath, dirname), n, l in zip(files, [3, 4], ['a', 'b']):
        g2, eta = list(map(float, dirname.split('_')))

        files_m = sorted([f for f in os.listdir(dirpath) if f.startswith(dirname)],
                         key=lambda x: -int(x.split('_')[-1]))
        for m, f in enumerate(files_m):
            # times = filedata[3].split('-')
            # times = np.linspace(0, float(times[0]), int(times[1]))
            levels = int(f.split('_')[-1])
            if levels == 50:
                continue

            filetrajs = os.listdir(os.path.join(dirpath, f))
            print(f'{len(filetrajs)} trajs found', f)
            # psuc = det_p_suc(times, g2, eta, D, n, method=method, row=1, adag=adag)
            # time_axs[n - 3].plot(times, psuc, c='k')

            # L = liouv_qvdp(1, g2, eta, D, n, n, m=50, adag=adag)
            # ev = L.eigenenergies(eigvals=5, sort='high')
            # t0, t1 = -1 / np.real(ev[n - 1]), -1 / np.real(ev[n])
            times, *_ = plot_memory_data(fig, time_axs[n - 3], n, n, os.path.join(dirpath, f), filetrajs,
                                         adag=adag, method=method, c=f'C{m}', label=levels)
        time_axs[n - 3].axhline(1 / n, c='k', ls='--')
        time_axs[n - 3].set(xlabel=r'$\gamma_1 t$', xscale='log')

        time_axs[n - 3].text(0.995, 0.97, rf'$({chr(97 + (n - 3))})\ n = {n}$', transform=time_axs[n - 3].transAxes,
                             ha='right', va='top', c='k', fontsize=10)

    # psuc = det_p_suc(times, g2, eta, D, n, method=method, row=1, adag=adag)
    # time_axs[n - 3].plot(times, psuc, c='k')
    # time_axs[n - 3].axvspan(t0, t1, color='grey', alpha=0.25, lw=0)
    handles, labels = time_axs[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], borderaxespad=0.02,
               ncol=len(labels), loc='upper center', title='Truncated Hilbert space dimension')
    for ax in time_axs:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    time_axs[0].set_xlabel('')
    fig.text(0.02, 0.5, 'Success probability', ha='center', va='center', rotation='vertical')

    fig.tight_layout(rect=[0.04, 0, 1, 0.825], pad=0.04)
    fig.savefig(local_plot_path(__file__) / f'memory_{"adag" if adag else "a"}_{method}_julia.pdf')


def plot_time_only_n(n, fig=None, ax=None, coh=True, method=1, adag=False):
    savefig = False
    if fig is None:
        latexify(plt, type='paper', fract=0.2, palette='qualitative')
        if ax is None:
            fig, ax = plt.subplots()
        savefig = True

    dirpath, dirname = _parse_files(n, adag)

    D = 0.4

    cmap = cm.get_cmap('Set1').reversed()

    g2, eta = list(map(float, dirname.split('_')))

    files_m = sorted([f for f in os.listdir(dirpath) if f.startswith(dirname)],
                     key=lambda x: -int(x.split('_')[-1]))
    for m, f in enumerate(files_m):
        # times = filedata[3].split('-')
        # times = np.linspace(0, float(times[0]), int(times[1]))
        levels = int(f.split('_')[-1])
        # if levels == 50:
        #     continue

        filetrajs = os.listdir(os.path.join(dirpath, f))
        print(f'{len(filetrajs)} trajs found', f)
        # psuc = det_p_suc(times, g2, eta, D, n, method=method, row=1, adag=adag)
        # time_axs[n - 3].plot(times, psuc, c='k')

        # L = liouv_qvdp(1, g2, eta, D, n, n, m=50, adag=adag)
        # ev = L.eigenenergies(eigvals=5, sort='high')
        # t0, t1 = -1 / np.real(ev[n - 1]), -1 / np.real(ev[n])
        times, *_ = plot_memory_data(fig, ax, n, n, os.path.join(dirpath, f), filetrajs,
                                     adag=adag, method=method, color=f'C{m}', label=levels)
    ax.axhline(1 / n, c='k', ls=':')
    ax.set(xlabel=r'$\gamma_1 t$', xscale='log', ylabel='Success probability')

    # time_ax.text(0.995, 0.97, rf'$(a)\ n = {n}$', transform=time_ax.transAxes,
    #              ha='right', va='top', c='k', fontsize=10)

    # psuc = det_p_suc(times, g2, eta, D, n, method=method, row=1, adag=adag)
    # time_axs[n - 3].plot(times, psuc, c='k')
    # time_axs[n - 3].axvspan(t0, t1, color='grey', alpha=0.25, lw=0)
    ax.legend(loc='lower left', title=r'$\dim \mathcal{H}_{eff}$', ncol=2)
    # handles, labels = time_ax.get_legend_handles_labels()
    # fig.legend(handles[::-1], labels[::-1], borderaxespad=0.02,
    #            ncol=len(labels), loc='upper center', title='Truncated Hilbert space dimension')
    # if time_ax.get_legend() is not None:
    #     time_ax.get_legend().remove()

    # fig.tight_layout(pad=0.04)
    if savefig:
        fig.savefig(local_plot_path(__file__, n, n) / f'memory_{n}_{"adag" if adag else "a"}_{method}_julia_single.pdf')

    return fig, ax


if __name__ == '__main__':
    # plot_decay_comparison()
    plot_time_only_n(4, method=1)
    # plot_time_only(False, method=0)
    # paper_plot(False)
