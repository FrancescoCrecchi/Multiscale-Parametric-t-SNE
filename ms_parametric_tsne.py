import numpy as np
from tqdm import tqdm
from parametric_tsne import ParametricTSNE


class MultiscaleParametricTSNE(ParametricTSNE):
    
    def __init__(self, n_components=2, n_iter=1000, early_exaggeration_epochs=50, early_exaggeration_value=4.0, early_stopping_epochs=np.inf, early_stopping_min_improvement=0.01, logdir=None, verbose=0):
        # Fake perplexity init.
        super().__init__(n_components=n_components, perplexity=-1.0, n_iter=n_iter, early_exaggeration_epochs=early_exaggeration_epochs, early_exaggeration_value=early_exaggeration_value, early_stopping_epochs=early_stopping_epochs, early_stopping_min_improvement=early_stopping_min_improvement, logdir=logdir, verbose=verbose)
    
    def _neighbor_distribution(self, X, tol=1e-4, max_iteration=50):
        # Compute multi-scale Gaussian similarities with exponentially growing perplexities
        N = X.shape[0]
        H = np.rint(np.log2(N/2))
        P = np.zeros((N, N))
        for h in tqdm(np.arange(1, H+1)):
            # Compute current perplexity P_ij
            perplexity = 2**h
            _P = self._compute_pij(X, perplexity, tol, max_iteration)
            
            # make symmetric and normalize
            _P += _P.T
            _P /= 2
            _P = np.maximum(_P, 1e-8)

            P += _P
        
        return P/H      # Average across perplexities