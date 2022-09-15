import os

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from ems import metastable_states
from utils import latexify, local_plot_path, local_data_path, amplitude


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


def plot_memory_data(fig, ax, n, m, dirpath, method=1, **ln_kw):
    times, P_amb, _, P_unamb, _ = success_probability(n, dirpath)

    ax.plot(times, P_amb, **ln_kw)
    # if up is not None:
    #     ax.fill_between(times, y1=p-up, y2=p+up, alpha=0.5, **ln_kw)

    ln_kw['label'] = None
    ax.plot(times, P_unamb, ls='--', **ln_kw)
    # if up is not None:
    #     ax.fill_between(times, y1=p2-up2, y2=p2+up2, alpha=0.5, **ln_kw)


def trace_distance(g2, eta, D, n, dim, initial_states):
    EMS, *_ = metastable_states(g2, eta, D, n, n, dim, g1=1)

    n = []
    for amp in initial_states:
        state = qt.coherent_dm(dim, amp)

        n.append(np.argmax([qt.tracedist(state, mu) for mu in EMS]))

    return np.array(n)


def success_probability(n, dirpath):
    filepath = str(dirpath) + '.npy'
    if os.path.exists(filepath):
        data = np.load(filepath)
        return [data[:, j] for j in range(5)]

    g1, g2, eta, D, dim = list(map(float, dirpath.name.split('_')))

    data = []
    initial_states = []
    for fname in dirpath.iterdir():
        if fname.name.endswith('.npy'):
            try:
                amp, phase = list(map(float, fname.name[:-4].split('_')))
                initial_states.append(amp * np.exp(1j * phase))
                data.append(np.real(np.load(fname)))
            except Exception as e:
                print(fname, e)
                fname.unlink()

    data = np.array(data)

    p_amb = data[:, :, 1:n + 1]

    overlaps = data[:, :, n + 1:]
    p_unamb = np.ones(p_amb.shape, dtype=complex)
    for k in range(n - 1):
        p_unamb[:, :, k] *= overlaps[:, :, k]
        for j in range(k + 1, n):
            p_unamb[:, :, j] *= (1 - overlaps[:, :, k])

    # beta = amplitude(g2, eta, n, n)
    # n = trace_distance(g2, eta, D, n, int(dim), np.array(initial_states))
    n = np.argmax(p_amb, axis=2)

    P_amp, UP_amb = _method_avg(p_amb, 1, n[:, 0])
    P_unamb, UP_unamb = _method_avg(p_unamb, 1, n[:, 0])

    times = data[0, :, 0]
    xxx = [times, P_amp, UP_amb, P_unamb, UP_unamb]

    save_data = np.concatenate(tuple(arr.reshape(-1, 1) for arr in xxx), axis=1)
    np.save(filepath, save_data)

    return xxx


def _parse_files(n, adag=True):
    dirpath = local_data_path(__file__, n, n)

    possible_files = [f for f in dirpath.iterdir() if f.isdir()]

    if len(possible_files) == 0:
        print('No file')
        return None
    elif len(possible_files) == 1:
        files = [possible_files[0]]
    else:
        print(f'Choose options for {n}:\n' + '\n'.join([f'{j}: {f.name}' for j, f in enumerate(possible_files)]))
        indexes = input('Which one? ')
        files = [possible_files[int(j)] for j in indexes.split(',')]

    return files


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


def plot_memory(n, files=None, fig=None, ax=None, labels=None, colors=None):
    if fig is None:
        latexify(plt, type='paper', fract=0.2, palette='qualitative')
        if ax is None:
            fig, ax = plt.subplots()

    paramfiles = files or _parse_files(n)

    for j, file in enumerate(paramfiles):
        g1, g2, eta, D, dim = list(map(float, file.name.split('_')))

        # L = eigvals(g2, eta, D, n, n, dim=50)
        # ev = L.eigenenergies(eigvals=5, sort='high')
        # t0, t1 = -1 / np.real(ev[n - 1]), -1 / np.real(ev[n])
        label = labels[j] if labels is not None else f'{int(dim)}'
        c = colors[j] if colors is not None else f'C{j}'
        plot_memory_data(fig, ax, n, n, file, method=1, color=c, label=label)

    ax.axhline(1 / n, c='k', ls=':')
    ax.set(xlabel=r'$\gamma_1 t$', xscale='log', ylabel='Success probability')

    return fig, ax


if __name__ == '__main__':
    memorypath = local_data_path('memory', 4, 4)
    dims = [15, 20, 30, 40]
    files = [memorypath / f'1_0.1_1.1423118881998557_0.4_{dim}' for dim in dims]

    fig, ax = plot_memory(4, files, labels=dims)
    fig.savefig(local_plot_path(__file__, 4, 4) / 'test.pdf')
