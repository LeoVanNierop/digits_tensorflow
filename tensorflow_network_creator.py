import json
from datetime import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class TensorflowNeuralNetwork(object):
    ACTIVATION_MAP = {
        "relu": tf.nn.relu,
        "softmax": tf.nn.softmax
    }
    OPTIMIZER_MAP = {
        "GradientDescent": tf.train.GradientDescentOptimizer,
        "Adam": tf.train.AdamOptimizer
    }
    OPTIMIZER_DECAY_MAP = {
        "exponential": tf.train.exponential_decay,
        "polynomial": tf.train.polynomial_decay
    }
    REDUCER_MAP = {
        "mean": tf.reduce_mean
    }
    LOSS_FUNCTION_MAP = {
        "softmax_cross_entropy": tf.nn.softmax_cross_entropy_with_logits
    }

    def _make_tf_layer(self, layer_info, in_variable, layer_name):
        bias = None
        if 'variables_shape' in layer_info:
            weights = tf.Variable(tf.truncated_normal(shape=layer_info['variables_shape'], stddev=0.1,
                                                      name=layer_name+"weights_seed"),
                                  name=layer_name+'weights')
            if layer_info.get('bias', False):
                bias = tf.Variable(tf.constant(0.1, shape=layer_info['out_shape'][1:], name=layer_name+"bias_seed"), name=layer_name+'bias')
        else:
            weights = None
        if layer_info['type'] == 'conv':
            strides = layer_info.get('strides', [1, 1, 1, 1])
            padding = layer_info.get('padding', 'SAME')
            output = tf.nn.conv2d(in_variable, weights, strides, padding=padding, name=layer_name+'output')
        elif layer_info['type'] == 'activation':
            output = self.ACTIVATION_MAP[layer_info['function']](in_variable, name=layer_name+'output')
        elif layer_info['type'] == 'pool':
            ksize = layer_info.get('ksize', [1, 2, 2, 1])
            strides = layer_info.get('strides', [1, 2, 2, 1])
            padding = layer_info.get('padding', 'SAME')
            output = tf.nn.max_pool(in_variable, ksize, strides, padding, name=layer_name+'output')
        elif layer_info['type'] == 'reshape':
            new_shape = [x if x is not None else -1 for x in layer_info['out_shape']]
            output = tf.reshape(in_variable, new_shape, name=layer_name+'output')
        elif layer_info['type'] == 'connected':
            output = tf.matmul(in_variable, weights, name=layer_name+'output')
        else:
            raise ValueError("Unknown layer type: {}".format(layer_info['type']))
        return {
            'weights': weights,
            'bias': bias,
            'output': output
        }

    def __init__(self, input_shape, output_shape, layers, train_settings, prediction_activation,
                 regularization=False, regularization_constant=None,
                 save_path="./model{}".format(datetime.now()),
                 op_name_prefix=""):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = layers
        self.train_settings = train_settings
        self.prediction_activation = prediction_activation
        self.regularization = regularization
        self.regularization_constant = regularization_constant
        self.save_path = save_path
        self.op_name_prefix = op_name_prefix

        self.layer_names = [self.op_name_prefix+"layer_{}_".format(i) for i in range(len(layers))]

        self.tf_global_step = tf.Variable(0, trainable=False, name=op_name_prefix+"global_step")
        self.tf_inputs = tf.placeholder(tf.float32, shape=layers[0]['in_shape'], name=op_name_prefix+"tf_inputs")
        self.tf_targets = tf.placeholder(tf.float32, shape=layers[-1]['out_shape'], name=op_name_prefix+"tf_targets")
        self.tf_layers = [None] * len(layers)
        in_variable = self.tf_inputs
        for i, layer in enumerate(self.layers):
            self.tf_layers[i] = self._make_tf_layer(layer, in_variable, self.layer_names[i])
            in_variable = self.tf_layers[i]['output']
        self.tf_regularization_base = tf.constant(0.0, shape=(1,), name=op_name_prefix+"regularization_base")
        if regularization:
            weights = [x['weights'] for x in self.tf_layers if x['weights'] is not None]
            if regularization_constant is None:
                self.tf_regularization_constant = tf.constant(1.0, dtype=tf.float32,
                                                           name=op_name_prefix+"regularization_constant")
            else:
                self.tf_regularization_constant = tf.constant(regularization_constant, dtype=tf.float32,
                                                           name=op_name_prefix + "regularization_constant")
            self.tf_regularization = tf.multiply(self.tf_regularization_constant,
                                                 tf.add_n([tf.norm(weight) for weight in weights]),
                                                 name=op_name_prefix+"regularization")

        self.tf_loss_base = self.REDUCER_MAP[train_settings['reducer']](
            self.LOSS_FUNCTION_MAP[train_settings['loss_function']](logits=self.tf_layers[-1]['output'],
                                                                    labels=self.tf_targets),
            name=op_name_prefix+"tf_loss_base")
        self.tf_loss = tf.add(self.tf_loss_base, self.tf_regularization/tf.cast(tf.shape(self.tf_inputs)[0],
                                                                                tf.float32,
                                                                                name=op_name_prefix+"loss_cast"),
                              name=op_name_prefix+"loss"
                              )

        optimizer = self.OPTIMIZER_MAP[train_settings['optimizer']]
        if train_settings['learning_rate'].get("decay_args") is not None:
            self.tf_learning_rate = self.OPTIMIZER_DECAY_MAP[train_settings[
                'learning_rate']["decay"]](global_step=self.tf_global_step,
                                           **train_settings['learning_rate']["decay_args"],
                                           name=op_name_prefix+"learning_rate")
        else:
            self.tf_learning_rate = tf.constant(train_settings['learning_rate']['base'],
                                                dtype=tf.float32,
                                                name=op_name_prefix + "learning_rate")

        self.tf_train_step = optimizer(self.tf_learning_rate).minimize(self.tf_loss,
                                                                       global_step=self.tf_global_step,
                                                                       name=op_name_prefix+"train_step")
        self.tf_label_prediction = tf.argmax(self.ACTIVATION_MAP[prediction_activation](self.tf_layers[-1]['output']),
                                             1,
                                             name=op_name_prefix+"label_prediction")
        self.tf_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.tf_label_prediction,
                                                               tf.argmax(self.tf_targets, 1))),
                                          name=op_name_prefix+"accuracy")
        self.tf_variable_initializer = tf.global_variables_initializer()
        self.tf_saver = tf.train.Saver(name=op_name_prefix+"saver")
        self.tf_sess = tf.Session()

    def store(self):
        init_vars = {
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'layers': self.layers,
            'train_settings': self.train_settings,
            'prediction_activation': self.prediction_activation,
            'regularization': self.regularization,
            'regularization_constant': self.regularization_constant,
            'save_path': self.save_path,
            'op_name_prefix': self.op_name_prefix
        }
        store_filename = self.save_path + "_meta.json"
        with open(store_filename, 'w') as f:
            f.write(json.dumps(init_vars))
        self.tf_saver.save(self.tf_sess, save_path=self.save_path+"model")

    @classmethod
    def load(cls, path):
        tf.reset_default_graph()
        store_filename = path + "_meta.json"
        with open(store_filename, 'r') as f:
            init_vars = json.loads(f.read())
        network = cls(**init_vars)
        network.tf_sess.run(network.tf_variable_initializer)
        network.tf_saver.restore(network.tf_sess, network.save_path+"model")
        return network

    def train(self, training_data, training_labels, batch_size, epochs, warm_start=False,
              test_data=None, test_labels=None):
        train_accuracy = []
        test_accuracy = []
        batches = int(training_data.shape[0]/batch_size)
        full_feed_dict = {self.tf_inputs: training_data, self.tf_targets: training_labels}
        test_feed_dict = {self.tf_inputs: test_data, self.tf_targets: test_labels}

        self.tf_sess.run(self.tf_variable_initializer)
        #self.tf_saver.save(self.tf_sess, save_path=self.save_path+"training", global_step=0)
        loss = "undefined"
        for epoch in range(epochs):
            print("starting epoch {}: {}".format(epoch, datetime.now()))
            print("\tloss: {}".format(loss))
            training_data, training_labels = shuffle(training_data, training_labels)
            print("\tstart training: {}".format(datetime.now()))
            for i in range(batches):

                feed_dict = {self.tf_inputs: training_data[(i*batch_size):((i+1)*batch_size), :],
                             self.tf_targets: training_labels[(i*batch_size):((i+1)*batch_size)]}
                _, loss = self.tf_sess.run([self.tf_train_step, self.tf_loss], feed_dict=feed_dict)
            if training_data.shape[0] > batch_size*batches:
                feed_dict = {self.tf_inputs: training_data[((i + 1) * batch_size):, :],
                             self.tf_targets: training_labels[((i + 1) * batch_size):]}
                _, loss = self.tf_sess.run([self.tf_train_step, self.tf_loss], feed_dict=feed_dict)
            print("\tdone training epoch {}: {}".format(epoch, datetime.now()))
            train_accuracy.append(self.tf_sess.run(self.tf_accuracy, feed_dict=full_feed_dict))
            if test_data is not None:
                test_accuracy.append(self.tf_sess.run(self.tf_accuracy, feed_dict=test_feed_dict))

        print("\tloss: {}".format(self.tf_sess.run(self.tf_loss, feed_dict=full_feed_dict)))
        #self.tf_saver.save(self.tf_sess, save_path=self.save_path+"training", global_step=1)

        return train_accuracy, test_accuracy

    def predict(self, test_data, save_path="d:/kaggle/digits/tf_model_saves/"):
        #this is broken because the session that trained ended and we lost the vars
        feed_dict = {self.tf_inputs: test_data}
        result = self.tf_sess.run(self.tf_label_prediction, feed_dict=feed_dict)
        return result


