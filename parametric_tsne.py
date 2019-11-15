import pathlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

from tqdm import tqdm
import numpy as np
import numba

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

from setGPU import setGPU
setGPU()

import keras.backend as K
from keras.models import Sequential
from keras.losses import kld
from keras.layers import Dense as fc, InputLayer
from keras.layers import Dropout
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf

import multiprocessing as mp

@numba.jit(nopython=True)
def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

@numba.jit(nopython=True)
def x2p_job(data, max_iteration=50, tol=1e-5):
    i, Di, logU = data
    
    beta = 1.0
    beta_min = -np.inf
    beta_max = np.inf

    H, thisP = Hbeta(Di, beta)
    Hdiff = H - logU

    tries = 0
    while tries < max_iteration and np.abs(Hdiff) > tol:
        thisP_old = thisP.copy()
    
        # If not, increase or decrease precision
        if Hdiff > 0:
            beta_min = beta
            if np.isinf(beta_max):      # Numba compatibility: isposinf --> isinf
                beta *= 2.
            else:
                beta = (beta + beta_max) / 2.
        else:
            beta_max = beta
            if np.isinf(beta_min):      # Numba compatibility: isneginf --> isinf
                beta /= 2. 
            else:
                beta = (beta + beta_min) / 2.

        H, thisP = Hbeta(Di, beta)
        if np.isnan(thisP).any():
            thisP = thisP_old.copy()
            break

        Hdiff = H - logU
        tries += 1

    return i, thisP

# @numba.jit(parallel=True)
def x2p(X, perplexity, n_jobs=None):        # Use all threads available

    n = X.shape[0]
    logU = np.log(perplexity)

    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X + (sum_X.reshape([-1, 1]) - 2 * np.dot(X, X.T))

    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape([n, -1])

    P = np.zeros([n, n])
    for i in range(n):
        P[i, idx[i]] = x2p_job((i,D[i],logU))[1]

    # def generator():
    #     for i in range(n):
    #         yield i, D[i], logU

    # with mp.Pool(n_jobs) as pool:
    #     result = pool.map(x2p_job, generator())
    # for i, thisP in result:
    #     P[i, idx[i]] = thisP

    return P

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


class ParametricTSNE(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=2, perplexity=30.,
                n_iter=1000,
                early_exaggeration_epochs = 50,
                early_exaggeration_value = 4.,
                early_stopping_epochs = np.inf,
                early_stopping_min_improvement = 1e-2,
                alpha = 1,
                nl1 = 500,
                nl2 = 500,
                nl3 = 2000,
                logdir=None, verbose=0):
        
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.verbose = verbose

        # FFNet architecture
        self.nl1 = nl1
        self.nl2 = nl2
        self.nl3 = nl3

        # Early-exaggeration
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.early_exaggeration_value = early_exaggeration_value
        # Early-stopping
        self.early_stopping_epochs = early_stopping_epochs
        self.early_stopping_min_improvement = early_stopping_min_improvement

        # t-Student params
        self.alpha = alpha

        # Tensorboard
        self.logdir = logdir

        # Internals
        self._model = None
        self._batch_size = None

        # Session handling
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(config=config, graph=self.graph)
        
    def fit(self, X, y=None, batch_size=None):

        # Setting session
        with self.graph.as_default():
            with self.sess.as_default():
                
                """fit the model with X"""
                n_sample, n_feature = X.shape

                self._batch_size = batch_size if batch_size is not None else n_sample

                self._log('Building model..', end=' ')
                self._build_model(n_feature, self.n_components)
                self._log('Done')

                self._log('Start training..')
                
                # Tensorboard
                callback = TensorBoard(self.logdir)
                callback.set_model(self._model)

                # Early stopping
                es_patience = self.early_stopping_epochs
                es_loss = np.inf
                es_stop = False
                
                epoch = 0
                while epoch < self.n_iter and not es_stop:
                    # Shuffle data and P as well!
                    new_indices = np.random.permutation(n_sample)
                    X = X[new_indices]

                    loss = 0.0
                    n_batches = 0
                    for i in range(0, n_sample, self._batch_size):

                        # Compute batch indices
                        batch_slice = slice(i, i + self._batch_size)

                        # Actual training
                        P_batch = self.calculate_P(X[batch_slice], self.perplexity)

                        # Early exaggeration        
                        if epoch < self.early_exaggeration_epochs:
                            P_batch *= self.early_exaggeration_value
                            
                        loss += self._model.train_on_batch(X[batch_slice], P_batch)

                        # Increase batch counter
                        n_batches += 1
                    
                    # End-of-epoch: summarize
                    loss /= n_batches

                    if epoch % 10 == 0:
                        self._log('Epoch: {0} - Loss: {1:.3f}'.format(epoch, loss))
                    # Write log
                    write_log(callback, ['loss'], [loss], epoch)

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

        with self.graph.as_default():
            with self.sess.as_default():
                """apply dimensionality reduction to X"""
                # fit should have been called before
                if self.model is None:
                    raise sklearn.exceptions.NotFittedError(
                        'This ParametricTSNE instance is not fitted yet. Call \'fit\''
                        ' with appropriate arguments before using this method.')

                self._log('Predicting embedding points..', end=' ')
                X_new = self.model.predict(X)
   
        self._log('Done')

        return X_new

    def fit_transform(self, X, y=None, batch_size=None):
        """fit the model with X and apply the dimensionality reduction on X."""
        self.fit(X, y, batch_size)

        X_new = self.transform(X)
        return X_new

    # ================================ Internals ================================

    def calculate_P(self, X, perplexity):
        n = X.shape[0]

        P = np.zeros([n, n])
        P = x2p(X, perplexity)
        P = P + P.T
        P = P / P.sum()
        P = np.maximum(P, 1e-8)
        
        return P

    def _kl_divergence(self, P, Y):
        sum_Y = K.sum(K.square(Y), axis=1)
        eps = K.variable(1e-15)
        D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
        Q = K.pow(1 + D / self.alpha, -(self.alpha + 1) / 2)
        Q *= K.variable(1 - np.eye(self._batch_size))
        Q /= K.sum(Q)
        Q = K.maximum(Q, eps)
        C = K.log((P + eps) / (Q + eps))
        C = K.sum(P * C)

        return C

    def _build_model(self, n_input, n_output):
        self._model = Sequential()
        self._model.add(InputLayer((n_input,)))
        # Layer adding loop
        for n in [self.nl1, self.nl2, self.nl3]:
            self._model.add(fc(n, activation='relu'))
        self._model.add(fc(n_output))
        self._model.compile('adam', self._kl_divergence)

    def _log(self, *args, **kwargs):
        """logging with given arguments and keyword arguments"""
        if self.verbose >= 1:
            print(*args, **kwargs)

    @property
    def model(self):
        return self._model
    


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
        '--logdir', type=str, default='None',
        help='where to store Tensorboard logs')

    main(parser.parse_args())
