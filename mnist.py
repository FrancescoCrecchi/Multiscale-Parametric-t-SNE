import os
import numpy as np
from scipy.io import loadmat

import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

from sklearn.manifold import TSNE

from secml.data.loader import CDataLoaderMNIST
from parametric_tsne import ParametricTSNE
from ms_parametric_tsne import MultiscaleParametricTSNE

def plot_mnist(X_2d, y, fname):
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c in zip(np.unique(y), colors):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=str(i), s=2)
    plt.legend()
    plt.savefig(fname)


if __name__ == "__main__":
    
    # Set parameters
    N = 10000
    BS = 5000
    N_EPOCHS = 100

    # Load data
    tr = loadmat('mnist_train.mat')
    np.random.RandomState(1234)
    ind = np.random.permutation(tr['train_X'].shape[0])
    # ind = np.arange(tr['train_X'].shape[0])
    train_X = tr['train_X'][ind[:N]]
    train_labels = tr['train_labels'][ind[:N]]

    X, y = train_X, train_labels.flatten() - 1      # Matlab classes are 1-based -> 0-based in Python
    print("Dataset size: {0}".format(X.shape[0]))

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
        n_iter=N_EPOCHS,
        verbose=1)

    # ptSNE = MultiscaleParametricTSNE(
    #     n_components=2,
    #     n_iter=N_EPOCHS,
    #     early_exaggeration_epochs=50,
    #     early_exaggeration_value=4.,
    #     early_stopping_epochs=np.inf,
    #     verbose=1)

    embds = ptSNE.fit_transform(X, batch_size=BS)
        
    OUTPUT_DIR = "output/mnist"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save output embds
    np.save(os.path.join(OUTPUT_DIR, 'mnist_ptnse_out.npy'), embds)

    # Plot
    plot_mnist(embds, y, os.path.join(OUTPUT_DIR, 'mnist_ptsne_plot.png'))

    print('done?')