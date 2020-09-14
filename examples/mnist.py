import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set()

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from msp_tsne import MultiscaleParametricTSNE

def plot(X_train, X_test, y_train, y_test):
    # Plot
    colors = cm.rainbow(np.linspace(0, 1, 10))
    # Train
    fig, ax = plt.subplots()
    ax.set_title('MNIST - Training scenario')
    for c in range(10):
        ax.scatter(X_train[y_train == c, 0], X_train[y_train == c, 1], s=8, color=colors[c], alpha=.6)
    fig.tight_layout()
    fig.show()
    # Extended
    fig, ax = plt.subplots()
    ax.set_title('MNIST - Extended scenario')
    X, y = np.vstack((X_train, X_test)), np.concatenate((y_train, y_test))
    for c in range(10):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=8, color=colors[c], alpha=.6)
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('msp_tsne', (MultiscaleParametricTSNE(n_components=2,
                                               n_iter=250,
                                               verbose=1)))
    ])

    # Fit
    X_tr_2d = pipe.fit_transform(X_train)

    # Transform
    X_ts_2d = pipe.transform(X_test)

    # Plot
    plot(X_tr_2d, X_ts_2d, y_train, y_test)