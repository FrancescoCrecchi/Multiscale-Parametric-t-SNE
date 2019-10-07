import numpy as np

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
    
    # Load data
    loader = CDataLoaderMNIST()
    tr = loader.load('training', num_samples=1000)

    # Normalize the features
    tr.X /= 255

    X, y = tr.X.get_data(), tr.Y.get_data()
    
    # Construct mappings using sklearn implementation
    sk_tSNE = TSNE(verbose=1)
    embds = sk_tSNE.fit_transform(X)

    # Save output embds
    np.save('mnist_sk_out.npy', embds)

    # Plot
    plot_mnist(embds, y, 'mnist_sk_plot.png')
    
    # Construct mappings        
    ptSNE = ParametricTSNE(verbose=1)
    embds = ptSNE.fit_transform(X)
    
    # Save output embds
    np.save('mnist_ptnse_out.npy', embds)

    # Plot
    plot_mnist(embds, y, 'mnist_ptsne_plot.png')

    print('done?')