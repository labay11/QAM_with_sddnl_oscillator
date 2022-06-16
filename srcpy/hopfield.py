import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from utils import latexify

class Hopfield:

    def __init__(self, N):
        self.N = N
        self.W = np.zeros((N, N), dtype=float)

    def train(self, data, method='hebb'):
        """Trains the network by constructing the weight matrix.

        Parameters
        ----------
        data : list of N-d arrays
            the training data, where each item represents a pattern
        """
        if method == 'pinv':
            W = self._train_pseudoinv(data)
        else:
            W = self._train_hebb(data)

        # set diag to 0
        self.W = W - np.diag(np.diag(W))

    def _train_hebb(self, data):
        return np.sum([np.outer(data[k, :], data[k, :]) for k in range(data.shape[0])], axis=0) / self.N

    def _train_pseudoinv(self, data):
        return data.T @ (data @ np.linalg.pinv(data)) @ data / self.N

    def _train_oja(self, data):
        # Simplified neuron model as a principal component analyzer [5]
        pass

    def predict(self, data, max_iter=10, eps=1e-6, threshold=0, flips=0, beta=0):
        """Trains the network by constructing the weight matrix.

        Parameters
        ----------
        data : N-d array
            the data to predict
        max_iter : int
            the maximum number of iterations
        eps : float
            the maximum error in the energy between flips before stopping.
        threshold : float
            the threshold in the Hopfield model that determines when a spin flips
        flips : int
            number of spin flips per iteration. A value of 0 determines that
            all spins are flipped synchronously.
        """
        assert flips >= 0
        assert eps >= 0
        assert -1 <= threshold <= 1

        s = np.copy(data)
        e = self.energy(s)

        if flips == 0:
            for _ in range(max_iter):
                s = np.sign(self.W @ s - threshold)
                _e = self.energy(s)

                yield s, _e

                if np.abs(_e - e) < eps:
                    break

                e = _e
        else:
            for _ in range(max_iter):
                to_flip = np.random.randint(0, self.N, flips)
                # s[to_flip] *= -1
                # coins = np.random.rand(flips)
                # _e = 2 * self.energy(s)
                # # print(np.exp(-beta * (_e - e)))
                # print(np.exp(-beta * (_e - e)), _e - e)
                # s[to_flip] *= np.where(coins < np.exp(-beta * (_e - e)), 1, -1)

                for j in to_flip:
                    s[j] = np.sign(self.W[j] @ s - threshold)

                _e = self.energy(s)
                #
                # print((_e - e))
                # if _e > e and np.random.rand() > np.exp(-beta * (_e - e)):
                #     s[j] *= -1

                yield s, _e, to_flip

                # if np.abs(_e - e) < eps:
                #     break

                e = _e

        return s, e

    def energy(self, s):
        """Calculates the energy of a given spin configuration."""
        return -0.5 * (s.T @ self.W @ s)


def save_time_ev(M, patterns):
    latexify(plt, type='beamer43', fract=0.08)
    N, frames = M.shape
    fig, ax = plt.subplots()
    mat = ax.matshow(M[:, 0].reshape(1, -1), cmap='binary', aspect='auto')
    ax.set(xlabel='Spins', xlim=(-0.5, N - 0.5), xticks=np.arange(0, N, 5), ylabel=r'$\uparrow/\downarrow$')
    ax.grid(False)
    ax.tick_params(axis='x', top=False, bottom=True, labelbottom=True, labeltop=False)
    ax.set_yticks([])

    def update(frame):
        mat.set_data(M[:, frame].reshape(1, -1))
        return mat,

    ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=100)
    ani.save('test.mp4', dpi=400)
    plt.show()


N = 50
hop = Hopfield(N)

patterns = np.array([
    [1] * N,
    [1] * (N//2) + [-1] * (N - N//2)
])

hop.train(patterns, method='hebb')

# a = np.random.randint(0, 2, N) * 2 - 1
a = np.array([1] * N)
a[np.random.rand(N) < 0.25] = -1
# print(a)

print('Overlaps')
for i, p in enumerate(patterns):
    print(i, ':', a.T @ p / N)

em = [a]

for i, (spins, e, flipped) in enumerate(hop.predict(a, flips=1, max_iter=100, beta=1)):
    # print('iter', i, 'conf', spins, 'e', e)
    em.append(spins.copy())

# cmap = LinearSegmentedColormap.from_list('ising', colors=[(-1, 'white'), (1, 'black'), (2, 'red')])
em = np.array(em)
np.save('spins.npy', -em.T)
M = -em.T if 1 < 0 else np.load('spins.npy')
save_time_ev(M, patterns)

print('Overlaps')
for i, p in enumerate(patterns):
    print(i, ':', em[-1].T @ p / N)
