from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy import stats
        
def update_weights(dense, lpm):
    def get_loss(m, x_shape):
            def loss(x):
                x = np.reshape(x, x_shape)
                l = m.predict_on_batch(x)
                return np.asscalar(l)
            return loss
    def get_gradients(m, x_shape):
        def gradient(input):
            # weights = [tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
            
            model_input = m.inputs[0]
            model_output = m.layers[-1].output
            
            out = K.transpose(K.gradients(model_output, model_input)[0])
            jacobian = K.function([model_input], [out])

            x = np.reshape(input, x_shape)
            x = jacobian([x])
            return np.asarray(x[0], dtype="float64", order='C').flatten()
        return gradient
    shape1 = np.expand_dims(gw(dense), 0).shape
    sol1 = scipy.optimize.minimize(get_loss(lpm, shape1), gw(dense).flatten(), jac=get_gradients(lpm, shape1), method="L-BFGS-B", options={'xtol': 1e-8, 'disp': True, 'maxcor': 100}).x
    sol1 = np.reshape(sol1, shape1)
    dense.set_weights([sol1[0, :-1], sol1[0, -1]])

# Custom metric returns prediction
def get_pred(y_true, y_pred):
    return y_pred

# Custom metrics returns label
def get_label(y_true, y_pred):
    return K.argmax(y_true)

# Custom loss returns absolute error between prediction and actual
def custom_loss(y_true, y_pred):
    return tf.math.abs(y_true - y_pred)


def zero_loss(y_true, y_pred):
    return K.sum(y_true * y_pred)

# Neural loss gets loss based on loss prediction model
def neural_loss(d):
    def loss(lpm, y_pred):
        weights = K.eval(y_pred[0, :-1])
        d.set_weights([weights, np.array(y_pred[0, -1])])
        l = lpm.predict_on_batch(y_pred)[0]
        return l
    return loss

# Returns combined and flattened weights and biases of layer
def gw(x): return np.concatenate([x.get_weights()[0],
                                  np.expand_dims(x.get_weights()[1].flatten(), 0)], 0)