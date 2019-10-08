import matplotlib as plt
plt.use('agg')
import matplotlib.pyplot as plt

import os
import numpy as np

if __name__ == "__main__":
    
    # Load results
    res = np.load(os.path.join('result', 'output.npy'))

    fig, ax = plt.subplots()
    ax.scatter(res[:, 0], res[:, 1])
    fig.savefig(os.path.join('result', 'output.png'))
