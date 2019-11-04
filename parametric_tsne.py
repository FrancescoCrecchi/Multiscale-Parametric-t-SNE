import pathlib
import os
import argparse

from tqdm import tqdm
import numpy as np

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

from setGPU import setGPU
setGPU()

import keras.backend as K
from keras.models import Sequential
from keras.losses import kld
from keras.layers import Dense as fc
from keras.layers import Dropout
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


# def scheduler(model, epoch):
#     if epoch < 250:
#         K.set_value(model.optimizer.momentum, 0.5)
#     else:
#         K.set_value(model.optimizer.momentum, 0.8)


class ParametricTSNE(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=2, perplexity=30.,
                n_iter=1000,
                early_exaggeration_epochs = 50,
                early_exaggeration_value = 4.,
                early_stopping_epochs = np.inf,
                early_stopping_min_improvement = 1e-2,
                logdir=None,
                verbose=0):
        """parametric t-SNE

        Keyword Arguments:

            - n_components -- dimension of the embedded space

            - perplexity -- the perplexity is related to the number of nearest
                            neighbors that is used in other manifold learning
                            algorithms

            - n_iter -- maximum number of iterations for the optimizaiton.

            - verbose -- verbosity level

            - logdir -- Tensorboard logdir (default: no log)
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.verbose = verbose

        # Early-exaggeration
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.early_exaggeration_value = early_exaggeration_value
        # Early-stopping
        self.early_stopping_epochs = early_stopping_epochs
        self.early_stopping_min_improvement = early_stopping_min_improvement
        # Tensorboard
        self.logdir = logdir

        # Internals
        self._model = None
        self._batch_size = None

    def fit(self, X, y=None, batch_size=None):
        """fit the model with X"""
        n_sample, n_feature = X.shape

        self._batch_size = batch_size if batch_size is not None else n_sample

        self._log('Building model..', end=' ')
        self._build_model(n_feature, self.n_components)
        self._log('Done')

        self._log('Start training..')
        
        # Tensorboard
        if self.logdir is not None:
            callback = TensorBoard(self.logdir)
            callback.set_model(self._model)

        # Precompute P once-for-all!
        P = self._neighbor_distribution(X)

        # Early stopping
        es_patience = self.early_stopping_epochs
        es_loss = np.inf
        es_stop = False
        
        # ------------ Actual training ------------
        epoch = 0
        while epoch < self.n_iter and not es_stop:
            # Shuffle data and P as well!
            new_indices = np.random.permutation(n_sample)
            X = X[new_indices]
            P = P[new_indices, :]
            P = P[:, new_indices]

            # Compute batching
            P_batches = self._compute_P_batches(P, self._batch_size)

            # Early exaggeration        
            if epoch < self.early_exaggeration_epochs:
                P_batches *= self.early_exaggeration_value

            loss = 0.0
            n_batches = 0
            for i in range(0, n_sample, self._batch_size):
                batch_slice = slice(i, i + self._batch_size)
                loss += self._model.train_on_batch(X[batch_slice], P_batches[batch_slice])
                # Increase batch counter
                n_batches += 1
            
            # End-of-epoch: summarize
            loss /= n_batches

            if epoch % 10 == 0:
                self._log('Epoch: {0} - Loss: {1:.3f}'.format(epoch, loss))
            
            # Tensorboard log
            if self.logdir is not None: write_log(callback, ['loss'], [loss], epoch)

            # Check early-stopping condition
            if loss < es_loss and np.abs(loss - es_loss) > self.early_stopping_min_improvement:
                es_loss = loss
                es_patience = self.early_stopping_epochs
            else:
                es_patience -= 1

            if es_patience == 0:
                self._log('Early stopping!')
                es_stop = True
            
            epoch += 1

        self._log('Done')

        return self  # scikit-learn does so..

    def transform(self, X):
        """apply dimensionality reduction to X"""
        # fit should have been called before
        if self._model is None:
            raise sklearn.exceptions.NotFittedError(
                'This ParametricTSNE instance is not fitted yet. Call \'fit\''
                ' with appropriate arguments before using this method.')

        self._log('Predicting embedding points..', end=' ')
        X_new = self._model.predict(X)
        self._log('Done')
        return X_new

    def fit_transform(self, X, y=None, batch_size=None):
        """fit the model with X and apply the dimensionality reduction on X."""
        self.fit(X, y, batch_size)

        X_new = self.transform(X)
        return X_new

    # ------------------ Internal methods ------------------

    def _compute_P_batches(self, P, batch_size=100):
        n = P.shape[0]
        P_batches = np.zeros(shape=(n, batch_size), dtype=np.float32)
        for i in range(0, n, batch_size):
            if i + batch_size > n:
                break

            batch_slice = slice(i, i + batch_size)
            P_batches[batch_slice, :] = P[batch_slice, batch_slice]

        return P_batches

    def _compute_pij(self, X, perplexity, tol=1e-4, max_iteration=50):
        """calculate neighbor distribution from X

        Keyword Arguments:

            - tol -- tolerance level for searching sigma numerically

            - max_iteration -- maximum number of iterations for finding sigma

        """
        n = X.shape[0]
        log_k = np.log(perplexity)

        # calculate squared l2 distance matrix D from X
        D = np.expand_dims(X, axis=0) - np.expand_dims(X, axis=1)
        D = np.square(D)
        D = np.sum(D, axis=-1)

        # find appropriate sigma values with bisection method and
        def beta_search(d_i):

            def Hbeta(D, beta):
                P = np.exp(-D * beta)           # TODO: Still numerical issues for large Ds...
                sumP = np.sum(P)
                H = np.log(sumP) + beta * np.sum(D * P) / sumP
                P /= sumP
                return H, P

            beta = 1.0
            beta_min = -np.inf
            beta_max = np.inf

            # Compute the gaussian kernel and entropy for the current precision
            H, thisP = Hbeta(d_i, beta)
            H_diff = H - log_k

            # Evaluate wheter the perplexity is within tolerance
            i = 0
            
            while i < max_iteration and np.abs(H_diff) > tol:
                thisP_old = thisP.copy()
                
                # If not, increase or decrease precision
                if H_diff > 0:
                    beta_min = beta
                    if np.isposinf(beta_max):
                        beta *= 2.
                    else:
                        beta = (beta + beta_max) / 2.
                else:
                    beta_max = beta
                    if np.isneginf(beta_min):
                       beta /= 2. 
                    else:
                        beta = (beta + beta_min) / 2.
                
                # Recompute the values
                H, thisP = Hbeta(d_i, beta)
                if np.isnan(thisP).any():
                    thisP = thisP_old.copy()
                    break

                H_diff = H - log_k
                i += 1

            return thisP

        P = np.zeros(shape=(n, n))
        for i in range(n):
            P[i, np.delete(np.arange(n), i)] = beta_search(np.concatenate((D[i, :i], D[i, i+1:])))

        # Optional free-up D
        del D
            
        return P/np.sum(P)          # TODO: CHECK THIS!

    def _neighbor_distribution(self, X, tol=1e-4, max_iteration=50):
        
        n = X.shape[0]
        P = self._compute_pij(X, self.perplexity, tol, max_iteration)
        
        # make P symmetric and normalize
        P = P + P.T
        P /= 2
        P = np.maximum(P, 1e-8)

        return P

    def _kl_divergence(self, P, Y):
        eps = K.variable(1e-14, dtype='float32')

        # calculate neighbor distribution Q (t-distribution) from Y
        D = K.expand_dims(Y, axis=0) - K.expand_dims(Y, axis=1)
        D = K.square(D)
        D = K.sum(D, axis=-1)

        Q = K.pow(1. + D, -1)

        # eliminate all diagonals
        non_diagonals = 1 - K.eye(self._batch_size, dtype='float32')
        Q = Q * non_diagonals

        # normalize
        sum_Q = K.sum(Q)
        Q /= sum_Q
        Q = K.maximum(Q, eps)

        divergence = K.sum(P * K.log((P + eps) / (Q + eps)))
        return divergence

    def _build_model(self, n_input, n_output):
        self._model = Sequential()
        self._model.add(fc(500, input_dim=n_input, activation='relu'))
        self._model.add(fc(500, activation='relu'))
        self._model.add(fc(2000, activation='relu'))
        self._model.add(fc(n_output))
        self._model.compile(Adam(), self._kl_divergence)      # optimizer=SGD(lr=200, momentum=0.5)

    def _log(self, *args, **kwargs):
        """logging with given arguments and keyword arguments"""
        if self.verbose >= 1:
            print(*args, **kwargs)


def main(args):
    from sklearn.preprocessing import StandardScaler

    RESULT_DIR = pathlib.Path('result')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print('Loading dataset.. ', end='')
    dataset = np.load(args.dataset).astype(np.float32)
    print('Done')

    # Scaling dataset
    dataset = StandardScaler().fit_transform(dataset)

    ptsne = ParametricTSNE(
        n_components=args.n_components,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        verbose=1,
        logdir=args.logdir)

    pred = ptsne.fit_transform(dataset)
    np.save(RESULT_DIR / 'output.npy', pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parametric t-SNE.')
    subparsers = parser.add_subparsers(dest='command')

    train_group = subparsers.add_parser(
        'train',
        description='Train a new parametric t-SNE model.')
    parser.add_argument(
        '--dataset', type=pathlib.Path,
        default=pathlib.Path('dataset', 'sample.npy'),
        help='dataset for training')
    parser.add_argument(
        '--n-components', type=int, default=2,
        help='dimension of embedded space')
    parser.add_argument(
        '--perplexity', type=float, default=30.,
        help='perplexity value')
    parser.add_argument(
        '--n-iter', type=int, default=1000,
        help='number of training epochs')
    parser.add_argument(
        '--logdir', type=str, default='.',
        help='where to store Tensorboard logs')

    main(parser.parse_args())
