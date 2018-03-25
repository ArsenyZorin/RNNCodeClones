import helpers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from random import random


class Seq2seq:

    def __init__(self, encoder_cell, decoder_cell, vocab_size, input_embedding_size, weights):
        self.scope = 'seq2seq_'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.encoder_cell = encoder_cell
            self.decoder_cell = decoder_cell
            self.vocab_size = vocab_size
            self.input_embedding_size = input_embedding_size
            self.weights = weights
            self.create_model()

        self.seq2seq_vars = tf.global_variables(self.scope)

    def create_model(self):
        self.create_placeholders()
        self.create_embeddings()
        self.init_encoder()
        self.init_decoder()
        self.init_optimizer()
        self.create_sess()

    def create_placeholders(self):
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

    def create_embeddings(self):
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1.0, 1.0), dtype=tf.float32)
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

    def init_encoder(self):
        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
            self.encoder_cell, self.encoder_inputs_embedded,
            dtype=tf.float32, time_major=True,)

    def init_decoder(self):
        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
            self.decoder_cell, self.decoder_inputs_embedded,
            initial_state=self.encoder_final_state,
            dtype=tf.float32, time_major=True, scope="plain_decoder",)

        self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.vocab_size)
        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

    def init_optimizer(self):
        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
            logits=self.decoder_logits,)

        self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def make_train_inputs(self, input_seq, target_seq):
        encoder_inputs_, _ = helpers.batch(input_seq)
        decoder_targets_, _ = helpers.batch(target_seq)
        decoder_inputs_, _ = helpers.batch(input_seq)
        return {
            self.encoder_inputs: encoder_inputs_,
            self.decoder_inputs: decoder_inputs_,
            self.decoder_targets: decoder_targets_,
        }

    def create_sess(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.embeddings.assign(self.weights))

    def train(self, length, vocab, batches, directory):

        help_batch = helpers.random_sequences(length_from=length['from'], length_to=length['to'],
                                              vocab_lower=vocab['lower'], vocab_upper=vocab['size'],
                                              batch_size=batches['size'])

        saver = tf.train.Saver(self.seq2seq_vars)
        loss_track = []
        try:
            path = directory
            for batch in range(batches['max'] + 1):
                seq_batch = next(help_batch)
                fd = self.make_train_inputs(seq_batch, seq_batch)
                _, loss, state = self.sess.run([self.train_op, self.loss, self.encoder_final_state[0]], fd)
                loss_track.append(loss)
                print('\rBatch {}/{}\tloss: {}\tshape: {}'.format(batch, batches['max'], loss, state.shape), end="")

                if batch == 0 or batch % batches['epoch'] == 0 or batch == batches['max']:
                    path = saver.save(self.sess, directory + '/seq2seq.ckpt', global_step=batch)

            plt.plot(loss_track)
            plt.savefig('plotfig.png')
            print('\nLoss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1],
                                                                         len(loss_track) * batches['size'],
                                                                         batches['size']))
            print("Trained model saved to {}".format(path))

        except KeyboardInterrupt:
            print('training interrupted')

    def restore(self, dir):
        saver = tf.train.Saver(self.seq2seq_vars)
        res, sess = helpers.load_model(saver, self.sess, dir)
        if res:
            self.sess = sess
            print('Model restored from {}'.format(dir))
            return self.sess
        else:
            return None

    def get_encoder_status(self, sequence):
        encoder_fs = []
        i = 1
        for seq in sequence:
            feed_dict = {self.encoder_inputs: [seq]}
            encoder_fs.append(self.sess.run(self.encoder_final_state[0], feed_dict=feed_dict))
            print('\r{}/{}'.format(i, sequence.size), end='')
            i += 1
        print()
        return encoder_fs

    def get_sess(self):
        return self.sess


class SiameseNetwork:
    def __init__(self, sequence_length, batch_size, layers):
        self.scope = 'siamese'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.input_x1 = tf.placeholder(tf.float32, shape=(None, sequence_length), name='originInd')
            self.input_x2 = tf.placeholder(tf.float32, shape=(None, sequence_length), name='cloneInd')
            self.input_y = tf.placeholder(tf.float32, shape=None, name='answers')

            self.sequence_length = sequence_length
            self.batch_size = batch_size
            self.layers = layers

            self.init_out()
            self.loss_accuracy_init()
            self.sess.run(tf.global_variables_initializer())

        self.siam_vars = tf.global_variables(self.scope)

    def init_out(self):
        self.out1 = self.rnn(self.input_x1, 'method1')
        self.out2 = self.rnn(self.input_x2, 'method2')
        self.distance = tf.sqrt(tf.reduce_sum(
            tf.square(tf.subtract(self.out1, self.out2))))
        self.distance = tf.div(self.distance,
                               tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1))),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.out2)))))

    def loss_accuracy_init(self):
        self.temp_sim = tf.subtract(tf.ones_like(self.distance, dtype=tf.float32),
                                    self.distance, name='temp_sim')
        self.correct_predictions = tf.equal(self.temp_sim, self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, 'float'), name='accuracy')

        self.loss = self.get_loss()
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def get_loss(self):
        tmp1 = (1 - self.input_y) * tf.square(self.distance)
        tmp2 = self.input_y * tf.square(tf.maximum(0.0, 1 - self.distance))
        return tf.add(tmp1, tmp2) / 2

    def rnn(self, input_x, name):
        with tf.name_scope('fw' + name), tf.variable_scope('fw' + name):
            stacked_rnn_fw = []
            for _ in range(self.layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.layers, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn_fw.append(fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope('bw' + name), tf.variable_scope('bw' + name):
            stacked_rnn_bw = []
            for _ in range(self.layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.layers, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn_bw.append(bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        with tf.name_scope('bw' + name), tf.variable_scope('bw' + name):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, [input_x], dtype=tf.float32)
        return outputs

    def dict_feed(self, x1_batch, x2_batch, y_batch=None):
        if random() > 0.5:

            if y_batch is not None:
                feed_dict = {
                    self.input_x1: x1_batch,
                    self.input_x2: x2_batch,
                    self.input_y: y_batch,
                }
            else:
                feed_dict = {
                    self.input_x1: x1_batch,
                    self.input_x2: x2_batch,
                }
        else:
            if y_batch is not None:
                feed_dict = {
                    self.input_x1: x2_batch,
                    self.input_x2: x1_batch,
                    self.input_y: y_batch,
                }
            else:
                feed_dict = {
                    self.input_x1: x2_batch,
                    self.input_x2: x1_batch,
                }
        return feed_dict

    def train(self, input_x1, input_x2, input_y, directory):
        batches = helpers.siam_batches(input_x1, input_x2, input_y)
        data_size = batches.shape[0]

        # saver = tf.train.Saver()
        # result, sess = helpers.load_model(saver, self.sess, directory + '/siam.ckpt')
        # if result:
        #     self.sess = sess
        #     print('model restored from {}'.format(directory))
        #     return self.sess

        print(data_size)
        saver = tf.train.Saver(self.siam_vars)
        save_path = directory
        for nn in range(data_size):

            x1_batch, x2_batch = helpers.shape_diff(batches[nn][0], batches[nn][1])
            y_batch = batches[nn][2]

            feed_dict = self.dict_feed(x1_batch, x2_batch, y_batch)
            _, loss, dist, temp_sim = \
                self.sess.run([self.train_op, self.loss, self.distance, self.temp_sim], feed_dict)
            # print('\rTRAIN: step {}/{} |\tloss {:g} |\t{} -- {}'.format(nn, data_size, loss, y_batch, dist), end="")
            print('TRAIN: step {}/{}\tloss {:g} |\tExpected: {}\tGot: {}'.format(nn, data_size, loss, y_batch, dist))
            # print('TRAIN: step {}, loss {:g}'.format(nn, loss))
            # print('EXPECTED: {}, GOT: {}'.format(y_batch, dist))
            if nn == 0 or nn % 1000 == 0 or nn == data_size - 1:
                save_path = saver.save(self.sess, directory + '/siam.ckpt', global_step=nn)

        print('Trained model saved to {}'.format(save_path))

    def restore(self, dir):
        saver = tf.train.Saver(self.siam_vars)
        res, sess = helpers.load_model(saver, self.sess, dir)
        if res:
            self.sess = sess
            print('Model restored from {}'.format(dir))
            return self.sess
        else:
            return None

    def eval(self, input_x1, input_x2=None, answ=None):
        if input_x2 is not None and answ is not None:
            eval_batches = np.asarray(list(zip(input_x1, input_x2, answ)))
            data_size = eval_batches.shape[0]

            eval_res = []
            step = 0
            for i in range(data_size):
                step += 1
                eval_res = self.step(eval_batches[i][0], eval_batches[i][1], eval_batches[i][2], step, eval_res)

            percentage = len(eval_res) / data_size
            print('Evaluation accuracy: {}'.format(percentage))

        elif input_x2 is None and answ is None:
            eval_batches = np.asarray(input_x1)
            data_size = eval_batches.shape[0]

            eval_res = []
            clones_list = []
            step = 0

            for i in range(data_size):
                for n in range(data_size, i + 1, -1):
                    if i == n:
                        continue
                    step += 1
                    eval_res += self.step(eval_batches[i], eval_batches[n], None, step, clones)

            percentage = len(eval_res) / data_size
            print('Clones percentage: {}'.format(percentage))
            return clones_list
        else:
            print('Invalid evaluation')
            sys.exit(1)

    def step(self, x1, x2, answ, step, clones):
        eval_res = []
        x1_batch, x2_batch = helpers.shape_diff(x1, x2)

        feed_dict = self.dict_feed(x1_batch, x2_batch)
        dist, sim = self.sess.run([self.distance, self.temp_sim], feed_dict)
        print('EVAL: step {}'.format(step))
        if answ is not None:
            print('Expected: {}\t Got {}:'.format(answ, dist))
            if int(x2[2]) == int(dist):
                eval_res.append(1)
        else:
            print('Answer: {}'.format(dist))
            if 1 == int(dist):
                eval_res.append(1)
                clones.append(x2)

        return eval_res

    #def eval(self, input_x1, input_x2, answ):
    #    if input_x2 is not None and answ is not None:
    #        # eval_batches = helpers.siam_batches(input_x1, input_x2, answ)
    #        eval_batches = np.asarray(list(zip(input_x1, input_x2, answ)))
    #        data_size = eval_batches.shape[0]

    #    # print(data_size)
    #        eval_res = []
    #        for nn in range(data_size):
    #            x1_batch, x2_batch = helpers.shape_diff(eval_batches[nn][0], eval_batches[nn][1])

    #            feed_dict = self.dict_feed(x1_batch, x2_batch)
    #            dist, _ = self.sess.run([self.distance, self.temp_sim], feed_dict)
    #            print('EVAL: step {}\t| Expected: {}\tGot: {}'.format(nn, eval_batches[nn][2], dist))
    #            # print('Expected: {}\t Got {}:'.format(eval_batches[nn][2], dist))
    #            if int(eval_batches[nn][2]) == int(dist):
    #                eval_res.append(1)

    #        percentage = len(eval_res) / data_size
    #        print('Evaluation accuracy: {}'.format(percentage))
    #    elif input_x2 is None and answ is None:
    #        eval_batches = np.asarray(input_x1)
    #        data_size = eval_batches.shape[0]

    #        eval_res = []
    #        clones_list = []

    #        for nn in range(data_size):
    #            x1_batch, x2_batch = helpers.shape_diff(x1,x2)

