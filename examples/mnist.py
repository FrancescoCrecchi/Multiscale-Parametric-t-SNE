import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set()

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from msp_tsne import MultiscaleParametricTSNE

if __name__ == '__main__':

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('msp_tsne', (MultiscaleParametricTSNE(n_components=2,
                                               n_iter=1000,
                                               verbose=1)))
    ])

    # Fit
    X_tr_2d = pipe.fit_transform(X_train)

    # Transform
    X_ts_2d = pipe.transform(X_test)

    # Plot
    fig, ax = plt.subplots()
    colors = cm.rainbow(np.linspace(0, 1, 10))
    for c in range(10):
        # Train
        ax.scatter(X_tr_2d[y_train == c, 0], X_tr_2d[y_train == c, 1], s=12, color=colors[c], alpha=.6)
        # # Test
        ax.scatter(X_ts_2d[y_test == c, 0], X_ts_2d[y_test == c, 1], s=6, color=colors[c], label=c)

    # Computing the limits of the axes
    X = np.vstack((X_tr_2d, X_ts_2d))
    xmin = X[:, 0].min()
    xmax = X[:, 0].max()
    expand_value = (xmax - xmin) * 0.05
    x_lim = np.asarray([xmin - expand_value, xmax + expand_value])

    ymin = X[:, 1].min()
    ymax = X[:, 1].max()
    expand_value = (ymax - ymin) * 0.05
    y_lim = np.asarray([ymin - expand_value, ymax + expand_value])

    fig.legend()
    fig.tight_layout()

    fig.show()