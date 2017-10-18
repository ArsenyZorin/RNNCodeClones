import numpy as np
import tensorflow as tf
import json
import sys
from model import Seq2seq
from random import random
import lstm_siamese

if len(sys.argv) < 2:
    print('Invalid usage of Seq2seq script')
    print('Please set directory with data')
    sys.exit(0)

if sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('Help message')
    sys.exit(0)

print('Arguments: {}'.format(sys.argv[1]))

tf.reset_default_graph()
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

print(tf.__version__)

weights_file = open(sys.argv[1] + 'pretrainedWeights', 'r')
weights = np.array(json.loads(weights_file.read()))

directory = 'trainedModel'

vocab_size = weights.shape[0]
vocab_lower = 2
vocab_upper = vocab_size

length_from = 1
length_to = 1000

batch_size = 1000
max_batches = 10
batches_in_epoch = 100

input_embedding_size = weights.shape[1]

layers = 5
encoder_hidden_units = layers
decoder_hidden_units = encoder_hidden_units

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

model = Seq2seq(encoder_cell, decoder_cell, vocab_size, input_embedding_size, weights)
model.train(length_from, length_to, vocab_lower, vocab_size,
     batch_size, max_batches, batches_in_epoch, directory)

origin_seq_file = open(sys.argv[1] + 'indiciesOriginCode', 'r')
orig_seq = np.array(json.loads(origin_seq_file.read()))

mutated_seq_file = open(sys.argv[1] + 'indiciesMutatedCode', 'r')
mutated_seq = np.array(json.loads(mutated_seq_file.read()))

nonclone_file = open(sys.argv[1] + 'indiciesNonClone', 'r')
nonclone_seq = np.array(json.loads(nonclone_file.read()))

# origin_encoder_states = model.get_encoder_status(np.append(orig_seq, orig_seq[:nonclone_seq.shape[0]]))
# clone_encoder_states = np.append(mutated_seq, nonclone_seq)
# answ = np.append(np.ones(orig_seq.shape[0]), np.zeros(nonclone_seq.shape[0]), axis=0)

origin_encoder_states = model.get_encoder_status(orig_seq[:200])
mutated_encoder_states = model.get_encoder_status(np.append(mutated_seq[:100], nonclone_seq[:100]))
answ = np.append(np.ones(100), np.zeros(100))


# LSTM RNN model
# _________________

lstm_model = lstm_siamese.LSTM(origin_encoder_states[0].shape[1], batch_size, layers)

# global_step = tf.Variable(0, name="global_step", trainable=False)
print("initialized siameseModel object")

# grads_and_vars = optimizer.compute_gradients(lstm_model.loss)
# tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
print("defined training_ops")

# grad_summaries = []
# for g, v in grads_and_vars:
#    if g is not None:
#        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
#        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
#        grad_summaries.append(grad_hist_summary)
#        grad_summaries.append(sparsity_summary)
# grad_summaries_merged = tf.summary.merge(grad_summaries)

loss_summary = tf.summary.scalar("loss", lstm_model.loss)
acc_summary = tf.summary.scalar("accuracy", lstm_model.accuracy)

# train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])

sess.run(tf.global_variables_initializer())

def train_step(x1_batch, x2_batch, y_batch, step):
    """
    A single training step
    """
    if random() > 0.5:
        feed_dict = {
            lstm_model.input_x1: x1_batch,
            lstm_model.input_x2: x2_batch,
            lstm_model.input_y: y_batch,
            lstm_model.dropout: 1.0,
        }
    else:
        feed_dict = {
            lstm_model.input_x1: x2_batch,
            lstm_model.input_x2: x1_batch,
            lstm_model.input_y: y_batch,
            lstm_model.dropout: 1.0,
        }
    loss, accuracy, dist = \
        sess.run([lstm_model.loss, lstm_model.accuracy, lstm_model.distance],  feed_dict)
    print("TRAIN: step {}, loss {:g}".format(step, loss))
    print(y_batch, dist)


batches = (origin_encoder_states, mutated_encoder_states, answ)
ptr = 0
max_validation_acc = 0.0

batches_size = len(batches[0])
for nn in range(batches_size):
    x1_batch = []
    x2_batch = []
    y_batch = batches[2][nn]

    size_diff = abs(batches[0][nn].shape[0] - batches[1][nn].shape[0])

    if batches[0][nn].shape[0] < batches[1][nn].shape[0]:
        x1_batch = np.append(batches[0][nn], np.zeros((size_diff, batches[0][nn].shape[1])), axis=0)
        x2_batch = batches[1][nn]
    else:
        x1_batch = batches[0][nn]
        x2_batch = np.append(batches[1][nn], np.zeros((size_diff, batches[1][nn].shape[1])), axis=0)

    train_step(x1_batch, x2_batch, y_batch, nn)
    # current_step = tf.train.global_step(sess, global_step)
    # sum_acc = 0.0
