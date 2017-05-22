import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow_network_creator import TensorflowNeuralNetwork
from datetime import datetime
from sklearn.utils import shuffle
import sys

print("imports done: {}".format(datetime.now()))

# LOADING THE TRAINING DATASET AND SPLITTING INTO TRAIN AND TEST

training_data = pd.read_csv('d:/kaggle/digits/train.csv')
def one_hot_digits(x):
    res = np.zeros((10,), dtype=float)
    res[x] = 1.0
    return(res)
labels = np.apply_along_axis(one_hot_digits, axis=1, arr=training_data[['label']])
images = training_data.loc[:, 'pixel0':'pixel783'].values.reshape(-1, 28, 28, 1)/255
flat_images = training_data.loc[:, 'pixel0':'pixel783'].values/255
labels, images, flat_images = shuffle(labels, images, flat_images)
test_proportion = 0.1
images_test = images[:int(test_proportion*images.shape[0])]
images_train = images[int(test_proportion*images.shape[0]):]
flat_images_test = flat_images[:int(test_proportion*images.shape[0])]
flat_images_train = flat_images[int(test_proportion*images.shape[0]):]
labels_test = labels[:int(test_proportion * images.shape[0])]
labels_train = labels[int(test_proportion * images.shape[0]):]

print("data loaded: {}".format(datetime.now()))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.5, global_step,
                                           images_train.shape[0]/20, 0.9, staircase=True)

convolution_model = {
    'input_shape': [None, 28, 28, 1],
    'output_shape':  [None, 10],
    'layers': [
        {
            'type': 'conv',
            'in_shape': [None, 28, 28, 1],
            'out_shape': [None, 28, 28, 8],
            'variables_shape': [5, 5, 1, 8],
            'bias': True
        },
        {
            'type': 'activation',
            'function': "relu"
        },
        {
            'type': 'pool',
            'in_shape': [None, 28, 28, 8],
            'out_shape': [None, 14, 14, 8],
            'ksize': [1, 2, 2, 1],
            'strides': [1, 2, 2, 1]
        },
        {
            'type': 'conv',
            'in_shape': [None, 14, 14, 8],
            'out_shape': [None, 14, 14, 8],
            'variables_shape': [5, 5, 8, 8],
            'bias': True
        },
        {
            'type': 'activation',
            'function': "relu"
        },
        {
            'type': 'reshape',
            'in_shape': [None, 14, 14, 8],
            'out_shape': [None, 14*14*8],
        },
        {
            'type': 'connected',
            'in_shape': [None, 14*14*8],
            'out_shape': [None, 10],
            'variables_shape': [14*14*8, 10],
            'bias': True
        }
    ],
    'train_settings': {
        'optimizer': "GradientDescent",
        'learning_rate': {
            "base": 0.5,
            "decay": "exponential",
            "decay_args": {
                "learning_rate": 0.5,
                "decay_steps": images_train.shape[0]/20,
                "decay_rate": 0.9,
                "staircase": True
            }
        },
        'loss_function': "softmax_cross_entropy",
        'reducer': "mean"
    },
    'prediction_activation': "softmax"
}


print("models defined: {}".format(datetime.now()))


# simple_nn_tf = Tensorflow_neural_network(**simple_nn_model)
# tr_acc_simple, tst_acc_simple = simple_nn_tf.train(flat_images_train,
#                                                    labels_train,
#                                                    100,
#                                                    1,
#                                                    test_data=flat_images_test,
#                                                    test_labels=labels_test)
#
# simple_nn_accuracies = pd.DataFrame({'training_accuracy': tr_acc_simple, 'test_accuracy': tst_acc_simple})

conv_nn_tf = TensorflowNeuralNetwork(**convolution_model,
                                     regularization=True,
                                     regularization_constant=0.1,
                                     save_path="d:/kaggle/digits/tf_model_saves/tf_save_load_test"
                                     )
tr_acc_conv, tst_acc_conv = conv_nn_tf.train(images_train,
                                             labels_train,
                                             100,
                                             2,
                                             test_data=images_test,
                                             test_labels=labels_test,
                                             )

conv_nn_tf.store()

conv_nn_accuracies = pd.DataFrame({'training_accuracy': tr_acc_conv, 'test_accuracy': tst_acc_conv})

submit_images = pd.read_csv('d:/kaggle/digits/test.csv').values.reshape(-1,28,28,1)/255
submit_example = pd.read_csv('d:/kaggle/digits/sample_submission.csv')
submit_example.Label = conv_nn_tf.predict(submit_images)

submit_example.to_csv("d:/kaggle/digits/save_load_test_output_results.csv", index=False)
