import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow_network_creator import Tensorflow_neural_network
from datetime import datetime
from sklearn.utils import shuffle

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
            'function': tf.nn.relu
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
            'function': tf.nn.relu
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
        'optimizer': tf.train.GradientDescentOptimizer(0.5),
        'loss_function': tf.nn.softmax_cross_entropy_with_logits,
        'reducer': tf.reduce_mean
    },
    'prediction_activation': tf.nn.softmax
}

simple_nn_model = {
    'input_shape': [None, 784],
    'output_shape':  [None, 10],
    'layers': [
        {
            'type': 'connected',
            'in_shape': [None, 784],
            'out_shape': [None, 784],
            'variables_shape': [784, 784],
            'bias': True
        },
        {
            'type': 'activation',
            'function': tf.nn.softplus
        },
        {
            'type': 'connected',
            'in_shape': [None, 784],
            'out_shape': [None, 10],
            'variables_shape': [784, 10],
            'bias': True
        }
    ],
    'train_settings': {
        'optimizer': tf.train.GradientDescentOptimizer(0.5),
        'loss_function': tf.nn.softmax_cross_entropy_with_logits,
        'reducer': tf.reduce_mean
    },
    'prediction_activation': tf.nn.softmax
}

print("models defined: {}".format(datetime.now()))

simple_nn_tf = Tensorflow_neural_network(**simple_nn_model)
tr_acc_simple, tst_acc_simple = simple_nn_tf.train(flat_images_train,
                                                   labels_train,
                                                   100,
                                                   10,
                                                   test_data=flat_images_test,
                                                   test_labels=labels_test)

simple_nn_accuracies = pd.DataFrame({'training_accuracy': tr_acc_simple, 'test_accuracy': tst_acc_simple})

conv_nn_tf = Tensorflow_neural_network(**convolution_model)
tr_acc_conv, tst_acc_conv = conv_nn_tf.train(images_train,
                                             labels_train,
                                             100,
                                             10,
                                             test_data=images_test,
                                             test_labels=labels_test)

conv_nn_accuracies = pd.DataFrame({'training_accuracy': tr_acc_conv, 'test_accuracy': tst_acc_conv})

submit_images = pd.read_csv('d:/kaggle/digits/test.csv').values.reshape(-1,28,28,1)
submit_example = pd.read_csv('d:/kaggle/digits/sample_submission.csv')
submit_example.Label = conv_nn_tf.predict(submit_images)

