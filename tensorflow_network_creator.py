from datetime import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class Tensorflow_neural_network(object):

    def _test_network_consistency(self):
        # TODO
        return True

    def _make_tf_layer(self, layer_info, in_variable):
        bias = None
        if 'variables_shape' in layer_info:
            weights = tf.Variable(tf.truncated_normal(shape=layer_info['variables_shape'], stddev=0.1))
            if layer_info.get('bias', False):
                bias = tf.Variable(tf.constant(0.1, shape=layer_info['out_shape'][1:]))
        else:
            weights = None
        if layer_info['type'] == 'conv':
            strides = layer_info.get('strides', [1,1,1,1])
            padding = layer_info.get('padding', 'SAME')
            output = tf.nn.conv2d(in_variable, weights, strides, padding=padding)
        elif layer_info['type'] == 'activation':
            output = layer_info['function'](in_variable)
        elif layer_info['type'] == 'pool':
            ksize = layer_info.get('ksize', [1, 2, 2, 1])
            strides = layer_info.get('strides', [1, 2, 2, 1])
            padding = layer_info.get('padding', 'SAME')
            output = tf.nn.max_pool(in_variable, ksize, strides, padding)
        elif layer_info['type'] == 'reshape':
            new_shape = [x if x is not None else -1 for x in layer_info['out_shape']]
            output = tf.reshape(in_variable, new_shape)
        elif layer_info['type'] == 'connected':
            output = tf.matmul(in_variable, weights)
        else:
            raise ValueError("Unknown layer type: {}".format(layer_info['type']))
        return {
            'weights': weights,
            'bias': bias,
            'output': output
        }

    def __init__(self, input_shape, output_shape, layers, train_settings, prediction_activation):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = layers
        self.inputs = tf.placeholder(tf.float32, shape=layers[0]['in_shape'])
        self.targets = tf.placeholder(tf.float32, shape=layers[-1]['out_shape'])
        if not self._test_network_consistency():
            raise ValueError("Network layers not consistent")
        self.tf_layers = [None] * len(layers)
        in_variable = self.inputs
        for i, layer in enumerate(self.layers):
            self.tf_layers[i] = self._make_tf_layer(layer, in_variable)
            in_variable = self.tf_layers[i]['output']
        self.tf_loss = train_settings['reducer'](train_settings['loss_function'](logits=self.tf_layers[-1]['output'],
                                                                                 labels=self.targets))
        self.train_step = train_settings['optimizer'].minimize(self.tf_loss)
        self.label_prediction = tf.argmax(prediction_activation(self.tf_layers[-1]['output']), 1)
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.label_prediction, tf.argmax(self.targets, 1))))
        self.variable_initializer = tf.global_variables_initializer()
        self.saver = tf.train.Saver()




    def train(self, training_data, training_labels, batch_size, epochs, warm_start=False,
              save_path="d:/kaggle/digits/tf_model_saves/", test_data=None, test_labels=None):
        train_accuracy = []
        test_accuracy = []
        batches = int(training_data.shape[0]/batch_size)
        full_feed_dict = {self.inputs: training_data, self.targets: training_labels}
        test_feed_dict = {self.inputs: test_data, self.targets: test_labels}
        with tf.Session() as sess:
            sess.run(self.variable_initializer)
            self.saver.save(sess, save_path=save_path, global_step=0)
            for epoch in range(epochs):
                print("starting epoch {}: {}".format(epoch, datetime.now()))
                print("\tloss: {}".format(sess.run(self.tf_loss, feed_dict=full_feed_dict)))
                training_data, training_labels = shuffle(training_data, training_labels)
                print("\tstart training: {}".format(datetime.now()))
                for i in range(batches):

                    feed_dict = {self.inputs: training_data[(i*batch_size):((i+1)*batch_size), :],
                                 self.targets: training_labels[(i*batch_size):((i+1)*batch_size)]}
                    sess.run(self.train_step, feed_dict=feed_dict)
                if training_data.shape[0] > batch_size*batches:
                    feed_dict = {self.inputs: training_data[((i + 1) * batch_size):, :],
                                self.targets: training_labels[((i + 1) * batch_size):]}
                    sess.run(self.train_step, feed_dict=feed_dict)
                print("\tdone training epoch {}: {}".format(epoch, datetime.now()))
                train_accuracy.append(sess.run(self.accuracy, feed_dict=full_feed_dict))
                if test_data is not None:
                    test_accuracy.append(sess.run(self.accuracy, feed_dict=test_feed_dict))

            print("\tloss: {}".format(sess.run(self.tf_loss, feed_dict=full_feed_dict)))
            self.saver.save(sess, save_path=save_path, global_step=1)

        return train_accuracy, test_accuracy



    def predict(self, test_data, save_path="d:/kaggle/digits/tf_model_saves/"):
        #this is broken because the session that trained ended and we lost the vars
        feed_dict = {self.inputs: test_data}
        with tf.Session() as sess:
            self.saver.restore(sess, save_path=self.saver.last_checkpoints[-1])
            result = sess.run(self.label_prediction, feed_dict=feed_dict)
        return result

if __name__ == '__main__':
    # make convnet for the digit classifier
    input_shape = [None, 28, 28, 1]
    output_shape = [None, 10]
    layers = [
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
    ]
    train_settings = {
        'optimizer': tf.train.GradientDescentOptimizer(0.5),
        'loss_function': tf.nn.softmax_cross_entropy_with_logits,
        'reducer': tf.reduce_mean
    }
    prediction_activation = tf.nn.softmax
    training_data = pd.read_csv('d:/kaggle/digits/train.csv')
    def one_hot_digits(x):
        res = np.zeros((10,), dtype=float)
        res[x] = 1.0
        return(res)
    labels = np.apply_along_axis(one_hot_digits, axis=1, arr=training_data[['label']])
    images = training_data.loc[:, 'pixel0':'pixel783'].values.reshape(-1, 28, 28, 1)/255
    images, labels = shuffle(images, labels)
    test_proportion = 0.1
    images_test = images[:int(test_proportion*images.shape[0])]
    images_train = images[int(test_proportion*images.shape[0]):]
    labels_test = labels[:int(test_proportion * images.shape[0])]
    labels_train = labels[int(test_proportion * images.shape[0]):]

    test_neural_net = Tensorflow_neural_network(input_shape, output_shape, layers, train_settings, prediction_activation)
    tr_acc, tst_acc = test_neural_net.train(images_train, labels_train, 100, 50, test_data=images_test, test_labels=labels_test)
    submission_set = pd.read_csv('d:/kaggle/digits/test.csv')
    submission_imgs = submission_set.values.reshape(-1, 28, 28, 1)/255
    submission_vals = test_neural_net.predict(submission_imgs)
    df = pd.read_csv("d:/kaggle/digits/sample_submission.csv")
    df.Label = submission_vals
    df.to_csv("d:/kaggle/digits/sub_tf_nn_extended.py", index=False)

