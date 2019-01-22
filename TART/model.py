"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: model.py
"""

import tensorflow as tf
import numpy as np
from .utils import check_dir, del_dir, create_dir
import os
from tensorflow.python.ops.rnn import dynamic_rnn

def transfer_params(from_scope, to_sope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_sope)

    trans_op = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        trans_op.append(to_var.assign(from_var))
    return tf.group(*trans_op)

def average_params(from_scopes, to_sope):
    for i, from_scope in enumerate(from_scopes):
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)

        if i == 0:
            from_varses = [[] for _ in range(len(vars))]

        for j in range(len(from_varses)):
            from_varses[j].append(vars[j])

    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_sope)

    trans_op = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_vars, to_var in zip(from_varses, to_vars):
        trans_op.append(to_var.assign(tf.add_n(from_vars) / len(from_scopes)))

    return tf.group(*trans_op)


from tensorflow.python.ops.rnn_cell_impl import LSTMCell, _WEIGHTS_VARIABLE_NAME, _BIAS_VARIABLE_NAME


class RELSTM(LSTMCell):
    def __init__(self, hidden_size, kernel_initializer=tf.initializers.random_normal,
                 bias_initializer=tf.keras.initializers.Zeros):
        super(RELSTM, self).__init__(hidden_size)
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    def build(self, inputs_shape):
        from tensorflow.python.ops import partitioned_variables
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        maybe_partitioner = (
            partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
            if self._num_unit_shards is not None
            else None)
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 4 * self._num_units],
            initializer=self._kernel_initializer,
            partitioner=maybe_partitioner)
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=self._bias_initializer)
        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                               initializer=self._initializer)

        if self._num_proj is not None:
            maybe_proj_partitioner = (
                partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
                if self._num_proj_shards is not None
                else None)
            self._proj_kernel = self.add_variable(
                "projection/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer,
                partitioner=maybe_proj_partitioner)
        self.built = True

class Model():

    def __init__(self, name=None):
        self.name = name

    # Layers
    ## Dense Layer
    def dense(self, bottom, name, in_size, out_size, use_relu=False, get_weight = False):
        with tf.variable_scope(name):
            weights, biases = self.get_dense_var_new(in_size, out_size)
            x = tf.reshape(bottom, [-1, in_size])
            output = tf.nn.bias_add(tf.matmul(x, weights), biases)
            if use_relu:
                output = tf.nn.relu(output)

            if get_weight:
                weight_dacay = tf.nn.l2_loss(weights, name='weight_dacay')
                return output, weight_dacay
            else:
                return output

    ## Conv Layer
    def conv2d(self, bottom, name, kernel_size=[3, 3], out_channel=512, stddev=0.01, use_relu=False, get_weight = False):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()[-1]
            filt = tf.get_variable(
                initializer=tf.random_normal([kernel_size[0], kernel_size[1], shape, out_channel], mean=0.0,
                                             stddev=stddev),
                name='filter')
            conv_biases = tf.get_variable(initializer=tf.zeros([out_channel]), name='biases')

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            output = tf.nn.bias_add(conv, conv_biases)

            if use_relu:
                output = tf.nn.relu(output)

            if get_weight:
                weight_dacay = tf.nn.l2_loss(filt, name='weight_dacay')
                return output, weight_dacay
            else:
                return output

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def get_dense_var_new(self, in_size, out_size):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, "weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, "biases")
        return weights, biases

    def get_var(self, initial_value, var_name):
        value = initial_value
        var = tf.get_variable(initializer=value, name=var_name)
        assert var.get_shape() == initial_value.get_shape()
        return var

    def GaussianNoise(self, x, sigma=0.01):
        x = x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma)
        return x

    def lstm_v2(self, bottom_squence, name, init_state=None):

        hidden_size = len(self.model_dict[name + '/rnn/lstm_cell/bias']) / 4

        lstm = RELSTM(hidden_size,
                      kernel_initializer=tf.constant_initializer(self.model_dict[name + '/rnn/lstm_cell/kernel']),
                      bias_initializer=tf.constant_initializer(self.model_dict[name + '/rnn/lstm_cell/bias'])
                      )

        if init_state is not None:
            state = init_state
        else:
            state = lstm.zero_state(bottom_squence.get_shape()[0], dtype=tf.float32)

        with tf.variable_scope(name):
            outputs, hidden_states = dynamic_rnn(lstm, bottom_squence, initial_state=state)
        hidden_states = hidden_states[0]

        return outputs, hidden_states

    def lstm(self, bottom_squence, name, seq_len, init_state=None, pre_forward_num=0):

        hidden_size = len(self.model_dict[name + '/lstm/relstm/bias']) / 4

        lstm = RELSTM(hidden_size,
                      kernel_initializer=tf.constant_initializer(self.model_dict[name + '/lstm/relstm/kernel']),
                      bias_initializer=tf.constant_initializer(self.model_dict[name + '/lstm/relstm/bias'])
                      )

        if init_state is not None:
            state = init_state
        else:
            state = lstm.zero_state(bottom_squence.get_shape()[0], dtype=tf.float32)

        outputs = []
        hidden_states = []

        with tf.variable_scope(name):
            for t in range(seq_len):
                output, state = lstm(bottom_squence[:, t, :], state)

                if t < pre_forward_num:
                    output = tf.stop_gradient(output)
                    state = (tf.stop_gradient(state[0]), tf.stop_gradient(state[1]))

                outputs.append(output)
                hidden_states.append(state[0])
        return outputs, hidden_states

    def lstm_new(self, bottom_squence, name, hidden_size, seq_len, init_state=None, pre_forward_num=0):

        lstm = RELSTM(hidden_size)
        self.lstm = lstm
        # init_state = None
        if init_state is not None:
            state = init_state
        else:
            state = lstm.zero_state(tf.shape(bottom_squence)[0], dtype=tf.float32)

        # state = lstm.zero_state(tf.shape(bottom_squence)[0], dtype=tf.float32)

        outputs = []
        hidden_states = []
        # print bottom_squence.get_shape()

        # print bottom_squence.get_shape().as_list()
        # exit()

        # input_tensors = []

        # for t in range(seq_len):
        #     input_tensors.append(bottom_squence[:,t,:])

        with tf.variable_scope(name):
            for t in range(seq_len):
                input_tensor = bottom_squence[:, t, :]
                # print tf.shape(input_tensor)
                # print input_tensor.get_shape()
                # exit()
                output, state = lstm(input_tensor, state)

                if t < pre_forward_num:
                    output = tf.stop_gradient(output)
                    state = (tf.stop_gradient(state[0]), tf.stop_gradient(state[1]))

                outputs.append(output)
                hidden_states.append(state)

        weight_decay = tf.nn.l2_loss(lstm._kernel)

        return outputs, hidden_states, weight_decay

    def lstm_new_v2(self, bottom_squence, name, hidden_size, init_state=None):
        lstm = RELSTM(hidden_size)

        if init_state is not None:
            state = init_state
        else:
            state = lstm.zero_state(bottom_squence.get_shape()[0], dtype=tf.float32)

        with tf.variable_scope(name):
            outputs, hidden_states = dynamic_rnn(lstm, bottom_squence, initial_state=state)

        hidden_states = hidden_states[0]

        weight_decay = tf.nn.l2_loss(lstm._kernel)

        return outputs, hidden_states, weight_decay

class Saver(object):
    def __init__(self,sess):
        self.sess = sess

    def get_variables(self , scope_name=None):
        vars = []
        with self.sess.as_default(), self.sess.graph.as_default():
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name):
                vars.append(var)
        return vars

    def load(self,model_path, scope_name = None, strict = False, show_names = False, del_scope = False):

        if type(model_path) == list:
            model_dict = {}
            for model_path0 in model_path:
                model_dict0 = np.load(model_path0, encoding='latin1').item()
                model_dict.update(model_dict0)
        else:
            model_dict = np.load(model_path, encoding='latin1').item()

        vars = self.get_variables(scope_name)

        ass_ops  =[]

        if show_names:
            print('var names:')
            for var in vars:
                print(var.name)
            print('loaded model names:')
            for key in model_dict:
                print(key)

        for var in vars:
            var_name = var.name
            if del_scope and scope_name is not None:
                if var_name.startswith(scope_name):
                    var_name = var_name[len(scope_name)+1:]
            if var_name in model_dict:
                ass_op = var.assign(model_dict[var_name])
                ass_ops.append(ass_op)

            else:
                assert not strict, "loaded model has no value of {}".format(var.name)

        self.sess.run(ass_ops)

    def save(self, save_path):
        params = {}
        with self.sess.as_default(), self.sess.graph.as_default():
            for var in tf.trainable_variables():
                # param_name = var.name.split(':')[0]
                param_name = var.name
                params[param_name] = self.sess.run(var)
        np.save(save_path, params)

    def _auto_save(self, save_path):
        params = {}
        for key in self.save_params:
            params[key] = self.sess.run(self.save_params[key])
        np.save(save_path, params)

    def auto_save_init(self, save_dir, save_interval, max_keep=5, save_scope_name=False, scope_name=None,
                       continue_train=False):
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.max_keep = max_keep

        self.model_step_dir = save_dir + '/steps/'

        if continue_train:
            assert check_dir(self.model_step_dir)
        else:
            if check_dir(self.model_step_dir):
                del_dir(self.model_step_dir)
            create_dir(self.model_step_dir)

        self.save_params = {}

        with self.sess.as_default(), self.sess.graph.as_default():
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name):
                # param_name = var.name.split(':')[0]
                param_name = var.name
                if not save_scope_name and scope_name is not None:
                    param_name = '/'.join(param_name.split('/')[1:])
                self.save_params[param_name] = var

    def auto_save(self, step):
        if step > 0 and step % self.save_interval == 0:
            self._auto_save( '{}/{}.npy'.format(self.model_step_dir, step))
            os.system('mv {} {}'.format(self.save_dir + 'init.npy', self.save_dir + 'init_bak.npy'))
            os.system('cp {}/{}.npy {}'.format(self.model_step_dir, step, self.save_dir + 'init.npy'))

            if step > self.max_keep * self.save_interval:
                rm_step = step - self.max_keep * self.save_interval
                os.system('rm {}/{}.npy'.format(self.model_step_dir, rm_step))