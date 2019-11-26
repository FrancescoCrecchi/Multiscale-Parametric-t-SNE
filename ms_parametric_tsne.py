import numpy as np
from tqdm import tqdm
from parametric_tsne import ParametricTSNE, x2p


class MultiscaleParametricTSNE(ParametricTSNE):
    
    def __init__(self, n_components=2,
                n_iter=1000,
                batch_size=500,
                early_exaggeration_epochs = 50,
                early_exaggeration_value = 4.,
                early_stopping_epochs = np.inf,
                early_stopping_min_improvement = 1e-2,
                nl1 = 1000,
                nl2 = 500,
                nl3 = 250,
                logdir=None, verbose=0):

        # Fake perplexity init.
        super().__init__(n_components=n_components,
                         perplexity=-1.0,       # !!!
                         n_iter=n_iter,
                         batch_size=batch_size,
                         early_exaggeration_epochs=early_exaggeration_epochs,
                         early_exaggeration_value=early_exaggeration_value,
                         early_stopping_epochs=early_stopping_epochs,
                         early_stopping_min_improvement=early_stopping_min_improvement,
                         nl1 = nl1,
                         nl2 = nl2,
                         nl3 = nl3,
                         logdir=logdir, verbose=verbose)
    
    def _calculate_P(self, X):
        # Compute multi-scale Gaussian similarities with exponentially growing perplexities
        n = X.shape[0]
        P = np.zeros([n, self.batch_size])
        H = np.rint(np.log2(n/2))
        for h in tqdm(np.arange(1, H+1)):
            # Compute current perplexity P_ij
            perplexity = 2**h
            for i in np.arange(0, n, self.batch_size):
                P_batch = x2p(X[i:i + self.batch_size], perplexity)
                P_batch[np.isnan(P_batch)] = 0
                P_batch = P_batch + P_batch.T
                P_batch = P_batch / P_batch.sum()
                P_batch = np.maximum(P_batch, 1e-12)
                P[i:i + self.batch_size] += P_batch
        
        return P/H      # Average across perplexities