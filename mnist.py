import numpy as np
from scipy.io import loadmat

import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

from sklearn.manifold import TSNE

from secml.data.loader import CDataLoaderMNIST
from parametric_tsne import ParametricTSNE

def plot_mnist(X_2d, y, fname):
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c in zip(np.unique(y), colors):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=str(i))
    plt.legend()
    plt.savefig(fname)


if __name__ == "__main__":
    
    # Set parameters
    N = 1000
    BS = 100

    # Load data
    tr = loadmat('mnist_train.mat')
    np.random.RandomState(1234)
    # ind = np.random.permutation(tr['train_X'].shape[0])
    ind = np.arange(tr['train_X'].shape[0])
    train_X = tr['train_X'][ind[:N]]
    train_labels = tr['train_labels'][ind[:N]]
    
    X, y = train_X, train_labels.flatten()

    # Perform PCA over the first 50 dimensions
    from sklearn.decomposition import PCA
    X = PCA(50).fit_transform(X)
    
    # # Construct mappings using sklearn implementation
    # sk_tSNE = TSNE(verbose=1)
    # embds = sk_tSNE.fit_transform(X)

    # # Save output embds
    # np.save('mnist_sk_out.npy', embds)

    # # Plot
    # plot_mnist(embds, y, 'mnist_sk_plot.png')
    
    # Construct mappings        
    ptSNE = ParametricTSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        early_exaggeration_epochs=50,
        early_exaggeration_value=4.,
        early_stopping_epochs=np.inf,
        verbose=1)

    embds = ptSNE.fit_transform(X, batch_size=BS)
    
    # Save output embds
    np.save('mnist_ptnse_out.npy', embds)

    # Plot
    plot_mnist(embds, y, 'mnist_ptsne_plot.png')

    print('done?')