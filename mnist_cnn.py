from __future__ import print_function
import os
import numpy as np
from setGPU import setGPU
setGPU(3)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.models import model_from_json

from sklearn.manifold import TSNE

# output classes
num_classes = 10

def train_cnn(X, y, input_shape=(28, 28, 1), batch_size=128, epochs=10):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Dense(num_classes, name='logits'))
    # model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(X, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.4)

    # save model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model


if __name__ == "__main__":

    from secml.data.loader import CDataLoaderMNIST

     # Load data
    loader = CDataLoaderMNIST()
    tr = loader.load('training', num_samples=10000)
    ts = loader.load('testing', num_samples=1000)

    # Normalize the features
    tr.X /= 255
    ts.X /= 255

    x_train, y_train = tr.X.get_data(), tr.Y.get_data()
    x_test, y_test = ts.X.get_data(), ts.Y.get_data()

    # reshape into images
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # classifier fit 
    if os.path.exists('model.json') and os.path.exists('model.h5'):
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk")
    #     model.compile(loss=keras.losses.categorical_crossentropy,
    #             optimizer=keras.optimizers.Adadelta(),
    #             metrics=['accuracy'])
    else:
        model = train_cnn(x_train, y_train)

    # # evaluate performance on test set
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # ===================================== T-SNE PART STARTS HERE! =====================================

    # Feature extractor
    # from keras.models import Model
    
    # feat_extr = Model(input=model.input, output=model.get_layer('logits').output)
    # feats = feat_extr.predict(x_test)

    # Compute embeddings using sklearn
    # from mnist import plot_mnist

    # y = y_test.argmax(axis=1)

    # sk_tSNE = TSNE(verbose=1)
    # embds = sk_tSNE.fit_transform(feats)
    # #  Save output embds
    # np.save('mnist_sk_feats_out.npy', embds)
    # # Plot
    # plot_mnist(embds, y, 'mnist_sk_feats_plot.png')

    # # Compute embeddings using ptSNE
    # from parametric_tsne import ParametricTSNE

    # ptSNE = ParametricTSNE(verbose=1)
    # embds = ptSNE.fit_transform(feats)
    # #  Save output embds
    # np.save('mnist_ptsne_feats_out.npy', embds)
    # # Plot
    # plot_mnist(embds, y, 'mnist_ptsne_feats_plot.png')

    # ---------------------- ADVERSARIAL EXAMPLES STARTS HERE! ----------------------
    import sys
    sys.path.insert(0, '/home/crecchi/AE_Detector')

    from generate_CW import generate_CW
    from utils import index_data, create_det_dset, mkdirs, plot_with_labels

    if not os.path.exists('X_CW.npy'):
        X   _CW = generate_CW(model, x_test, y_test, '.', n_samples=100)
    else:
        X_CW = np.load('X_CW.npy', allow_pickle=True).item()

    X_NAT = index_data(x_test, y_test)
    ds = create_det_dset(X_NAT, X_CW, num_classes)

    for T in range(num_classes):
        m = min(X_NAT[T].shape[0], X_CW[T].shape[0])
        nat_embds = feat_extr.predict(X_NAT[T])[:m]
        adv_embds = feat_extr.predict(X_CW[T])[:m]

        feats = np.concatenate((nat_embds, adv_embds))

        # Compute embeddings
        
        sk_tSNE = TSNE(verbose=1)
        embds = sk_tSNE.fit_transform(feats)
        # ptSNE = ParametricTSNE(verbose=1)
        # embds = ptSNE.fit_transform(feats)
    
        lbls = np.zeros(embds.shape[0])
        lbls[m:] = 1        # Adv

        fig, ax = plt.subplots()
        plot_with_labels(ax, embds, lbls, ['natural', 'adversarial'], title=T)
        fig.savefig(mkdirs(os.path.join('adv_plots', str(T))))