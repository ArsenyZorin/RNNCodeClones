import helpers
import tensorflow as tf
import matplotlib.pyplot as plt


class Seq2seq:

    PAD = 0
    EOS = 1

    def __init__(self, encoder_cell, decoder_cell, vocab_size, input_embedding_size, weights):
        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell
        self.vocab_size = vocab_size
        self.input_embedding_size = input_embedding_size
        self.sess = tf.InteractiveSession()
        self.weights = weights
        self.create_model()

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
        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
            logits=self.decoder_logits,)

        self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def make_train_inputs(self, input_seq, target_seq):
        self.encoder_inputs_, _ = helpers.batch(input_seq)
        self.decoder_targets_, _ = helpers.batch(target_seq)
        self.decoder_inputs_, _ = helpers.batch(input_seq)
        return {
            self.encoder_inputs: self.encoder_inputs_,
            self.decoder_inputs: self.decoder_inputs_,
            self.decoder_targets: self.decoder_targets_,
        }

    def create_sess(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.embeddings.assign(self.weights))

    def train(self, length_from,
              length_to, vocab_lower,
              vocab_upper, batch_size,
              max_batches, batches_in_epoch, directory):

        batches = helpers.random_sequences(length_from=length_from, length_to=length_to,
                                           vocab_lower=vocab_lower, vocab_upper=vocab_upper,
                                           batch_size=batch_size)

        result, smth = helpers.load_model(directory, self.sess)
        if result:
            self.sess = smth
            seq_batch = next(batches)
            loss = self.sess.run(self.loss, self.make_train_inputs(seq_batch, seq_batch))
            print('model restored from {}'.format(directory))
            print('model less: {}'.format(loss))
            return self.sess

        loss_track = []
        try:
            for batch in range(max_batches + 1):
                seq_batch = next(batches)
                fd = self.make_train_inputs(seq_batch, seq_batch)
                _, l = self.sess.run([self.train_op, self.loss], fd)
                loss_track.append(l)
                current_loss = self.sess.run(self.loss, fd)
                print('\rBatch ' + str(batch) + '/' + str(max_batches) + ' loss: ' + str(current_loss), end="")

                if batch == 0 or batch % batches_in_epoch == 0:
                    print('\nbatch {}'.format(batch))
                    print('  minibatch loss: {}'.format(current_loss))
                    predict_ = self.sess.run(self.decoder_prediction, fd)
                    for i, (inp, pred) in enumerate(zip(fd[self.encoder_inputs].T, predict_.T)):
                        print('  sample {}:'.format(i + 1))
                        print('    input     > {}'.format(inp))
                        print('    predicted > {}'.format(pred))
                        if i >= 2:
                            break
                    print()

            plt.plot(loss_track)
            plt.savefig('plotfig.png')
            print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1],
                                                                         len(loss_track) * batch_size, batch_size))

            helpers.save_model(directory, self.sess)
            print("Trained model saved to {}".format(directory))

        except KeyboardInterrupt:
            print('training interrupted')

    def get_encoder_status(self, sequence):
        encoder_fs = []
        for seq in sequence:
            feed_dict = {self.encoder_inputs: [seq]}
            encoder_fs.append(self.sess.run(self.encoder_final_state[0], feed_dict=feed_dict))

        return encoder_fs

    def get_sess(self):
        return self.sess

