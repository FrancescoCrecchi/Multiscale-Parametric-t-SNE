
# Multiscale Parametric t-SNE
Reference implementation for the paper: ["Perplexity-free Parametric t-SNE"](#).

Multiscale extension of parametric t-SNE which relieves the user from tuning the `perplexity` parameter (either by hand or via cross-validation).
This implementation exploits [keras](https://keras.io/) to provide GPU acceleration during model training and inference, while maintaining a [scikit-learn](https://scikit-learn.org/)
compatible interface that allows to use `MultiscaleParamerticTSNE` as part of a [pipeline](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) replacing the library t-SNE implementation.

In addition to the perplexity-free model, a refined `ParamerticTSNE` model is released.
As for the multiscale implementation, it favours of GPU acceleration for neural network training and inference and is sklearn compatible. This allows the user to search for the best perplexity parameter using `sklearn.model_selection.GridSearchCV` module, for example.      

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This program was tested under Python 3.6. All the required packages are contained in `setup.py`

### Installing

After cloning this repository, install the package by running the following:

```
pip3 install .
```

## Deployment

Simply create a `ParametricTSNE` or `MultiscaleParametricTSNE` instance. The interface was designed similarly to that of scikit-learn estimators.

```python
from msp_tsne import MultiscaleParametricTSNE

transformer = MultiscaleParametricTSNE()

# suppose you have the dataset X
X_new = transformer.fit_transform(X)

# transform new dataset X2 with pre-trained model
X2_new = transformer.transform(X2)
```

## Built With

- [scikit-learn](http://scikit-learn.org/stable/) - Extensive machine learning framework

- [Keras](https://keras.io) - Deep learning framework wrapper that supports TensorFlow, Theano, and CNTK

## Authors

- __Francesco Crecchi__ - Research and implementation - [FrancescoCrecchi](https://github.com/FrancescoCrecchi)

## Acknowledgements

- This project was forked from [zaburo-ch's implementation](https://github.com/zaburo-ch/Parametric-t-SNE-in-Keras).
