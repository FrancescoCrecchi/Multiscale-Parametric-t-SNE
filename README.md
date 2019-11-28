
# Multiscale Parametric t-SNE

multiscale extension of parametric t-SNE which relieves the user from tuning the `perplexity` parameter (either by hand or via cross-validation). It is proven to better retain both the local and the global data structure than original parametric t-SNE.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This program was tested under Python 3.7. All the required packages are contained in `setup.py`

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
