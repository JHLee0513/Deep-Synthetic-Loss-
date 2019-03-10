from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Conv2D, Reshape, MaxPooling2D, Flatten, ReLU, Activation, AlphaDropout
from keras.layers import Lambda, Dropout, Input, GlobalAveragePooling2D, BatchNormalization, Add
from keras.activations import selu
from keras.utils import np_utils
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adamax
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy import stats
import helpers

# ========= DATA PREPROCESSING STEP =============
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# ========= DATA PREPROCESSING STEP =============

# ========= ORIGINAL MODEL =============
input_tensor = Input((28, 28, 1))
f1 = Flatten()(input_tensor)
# f1 = input_tensor
d1 = Dense(250)
a1 = Activation(activation="relu")((d1)(f1))
d2 = Dense(50)
a2 = Activation(activation="relu")((d2)(a1))
# a2 = Flatten()(a2)
# last softmax layer
d3 = Dense(units=10)
a3 = Activation(activation="softmax")((d3)(a2))
model = Model(inputs=input_tensor, outputs=a3)
# ========= ORIGINAL MODEL =============

# ========= LOSS PREDICTION MODEL =============
# Small SNN ("Self-Normalizing Neural Networks "https://arxiv.org/pdf/1706.02515.pdf)
def lpm(input_shape):
    input = Input(input_shape)
    f = Flatten()
    x = Dense(256, activation=selu, kernel_initializer='lecun_normal')(f(input))
    # x = AlphaDropout(.05)(x)
    latent = Dense(128, activation=selu, kernel_initializer='lecun_normal')(x)
    # x = AlphaDropout(.05)(x)
    loss_pred = Dense(1, kernel_initializer='lecun_normal')(latent)
    # latent_input = Input((128,))
    # x = Dense(128, activation=selu, kernel_initializer='lecun_normal')(latent_input)
    # weights = Dense(input_shape[0] * input_shape[1], kernel_initializer='lecun_normal')(x)
    # weights = Reshape(input_shape)(weights)
    lpm = Model(inputs=input, outputs=loss_pred)
    # lom = Model(inputs=latent_input, outputs=weights)
    return lpm #, lom

# ========= LOSS PREDICTION MODEL =============


# Compile original model
model.compile(loss='categorical_crossentropy', optimizer=Adamax())

# Instantiate loss models for layers 1-3 of original model
lpm1 = lpm(helpers.gw(d1).shape)
lpm2 = lpm(helpers.gw(d2).shape)
lpm3 = lpm(helpers.gw(d3).shape)

# Use Adadelta optimizer for sparse inputs
lpm1.compile(loss=helpers.custom_loss, optimizer=Adadelta(), metrics=[helpers.get_pred])
lpm2.compile(loss=helpers.custom_loss, optimizer=Adadelta(), metrics=[helpers.get_pred])
lpm3.compile(loss=helpers.custom_loss, optimizer=Adadelta(), metrics=[helpers.get_pred])


# lom1.compile(loss=neural_loss(d1), optimizer=Adadelta())
# lom2.compile(loss=neural_loss(d2), optimizer=Adadelta())
# lom3.compile(loss=neural_loss(d3), optimizer=Adadelta())

# ========= TRAINING =============
def loss_optimization(lpm):
    def opt(x):
        return lpm.predict_on_batch(x)
    return opt

BATCH_SIZE = 128
EPOCHS = 15
# TRUE_LOSS = []
# PRED_LOSS = [[], [], []]
for e in range(EPOCHS):
    for i in range(len(X_train) // BATCH_SIZE):
        loss = model.evaluate(X_train[i * BATCH_SIZE: (i+1) * BATCH_SIZE],
                              y_train[i * BATCH_SIZE: (i+1) * BATCH_SIZE], steps=1)
        loss = np.expand_dims(loss, 0)
        print(loss)
        # Last layer loss prediction learns off of true signal
        # Other layers learn off of last layer's loss prediction
        # Similar approach used in 'Decoupled Neural Interfaces
        # using Synthetic Gradients': https://arxiv.org/pdf/1608.05343.pdf
        loss3 = lpm3.train_on_batch(np.expand_dims(helpers.gw(d3), 0), loss)
        loss2 = lpm2.train_on_batch(np.expand_dims(
            helpers.gw(d2), 0), np.expand_dims(loss3[1], 0))
        loss1 = lpm1.train_on_batch(np.expand_dims(
            helpers.gw(d1), 0), np.expand_dims(loss2[1], 0))

        # Change model weights
        if e > 1:
            helpers.update_weights(d1, lpm1)
            helpers.update_weights(d2, lpm2)
            helpers.update_weights(d3, lpm3)

        # if e > 13:
        #     TRUE_LOSS.append(loss)
        #     PRED_LOSS[0].append(loss1[1])
        #     PRED_LOSS[1].append(loss2[1])
        #     PRED_LOSS[2].append(loss3[1])


        
        print("Loss 1: " + str(loss1) + "\tLoss 2: " +
              str(loss2) + "\tLoss 3: " + str(loss3))
# ========= TRAINING =============

# # ========= DATA VIS =============
# for loss_met in PRED_LOSS:
#     slope, intercept, r_value, p_value, std_err = stats.linregress(
#         np.array(TRUE_LOSS).flatten(), loss_met)
#     print("R:" + str(r_value) + "\tP:" + str(p_value))
#     line = np.array(TRUE_LOSS) * slope + intercept
#     plt.scatter(TRUE_LOSS, loss_met, s=2.0)
#     plt.plot(TRUE_LOSS, line)
#     plt.show()
