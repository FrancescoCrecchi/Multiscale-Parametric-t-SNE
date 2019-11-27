import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from msp_tsne import ParametricTSNE, MultiscaleParametricTSNE


def plot_embeds(X_2d, y):
    plt.figure()
    classes = np.unique(y)
    for c in classes:
        plt.scatter(X_2d[y == c, 0], X_2d[y == c, 1], s=2, label=c)
    plt.legend()
    plt.show()


if __name__ == '__main__':

    X, y = load_digits(return_X_y=True)
    X /= 255.

    ptsne = ParametricTSNE(n_components=2, perplexity=30, verbose=1)
    # ptsne = MultiscaleParametricTSNE(n_components=2, verbose=1)
    
    X_2d = ptsne.fit_transform(X)
    plot_embeds(X_2d, y)