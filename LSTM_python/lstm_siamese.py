import numpy as np
import tensorflow as tf

class LSTM:

    def __init__(self, input_x1, input_x2, embedding_size, sequence_length, batch_size):
        self.input_x1 = input_x1
        self.input_x2 = input_x2

        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.input_y = tf.placeholder(tf.float32, [None])
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        self.init_out()


    def init_out(self):
        self.out1 = self.rnn(self.input_x1, 'method1')
        self.out2 = self.rnn(self.input_x2, 'method2')
        self.distance = tf.sqrt(tf.reduce_sum(
            tf.square(tf.substr(self.out1, self.out2)), 1, keep_dims=True))
        self.distance = tf.div(self.distance,
                               tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
        self.distance = tf.reshape(self.distance, [-1], name="distance")


    def loss_accuracy_init(self):
        self.loss = get_loss()

        self.temp_sim = tf.subtract(tf.ones_like(self.distance),
                                    tf.rint(self.distance), name="temp_sim")
        self.correct_predictions = tf.equal(self.temp_sim, self.input_y)
        self.accuracy=tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

    def get_loss(self):
        tmp = self.input_y * tf.square(self.distance)
        tmp2 = (1 - self.input_y) * tf.square(tf.maximum((1 - self.distance),0))
        return tf.reduce_sum(tmp + tmp2) / self.batch_size / 2


    def rnn(self, input_x, name):
        self.layers = 5
        with tf.name_scope('fw' + name), tf.variable_scope('fw' + name):
            stacked_rnn_fw = []
            for _ in range(self.layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.sequence_length, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope('bw' + name), tf.variable_scope('bw' + name):
            stacked_rnn_bw = []
            for _ in range(self.layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.sequence_length, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        # Get lstm cell output

        with tf.name_scope('bw' + name), tf.variable_scope('bw' + name):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, input_x, dtype=tf.float32)
        return outputs[-1]









