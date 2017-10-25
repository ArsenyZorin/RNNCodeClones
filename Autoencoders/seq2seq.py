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
# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()

print(tf.__version__)

weights_file = open(sys.argv[1] + 'pretrainedWeights', 'r')
weights = np.array(json.loads(weights_file.read()))

directory = sys.argv[1] + '/trainedModel'

vocab_size = weights.shape[0]
vocab_lower = 2
vocab_upper = vocab_size

length_from = 1
length_to = 1000

batch_size = 100
max_batches = 1000 # 20000
batches_in_epoch = 1000

input_embedding_size = weights.shape[1]

layers = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

model = Seq2seq(encoder_cell, decoder_cell, vocab_size, input_embedding_size, weights)
model.train(length_from, length_to, vocab_lower, vocab_size,
     batch_size, max_batches, batches_in_epoch, directory)

origin_seq_file = open(sys.argv[1] + 'indiciesOriginCode', 'r')
orig_seq = np.array(json.loads(origin_seq_file.read()))

eval_seq_file = open(sys.argv[1] + 'EvalCode', 'r')
eval_seq = np.array(json.loads(eval_seq_file.read()))

mutated_seq_file = open(sys.argv[1] + 'indiciesMutatedCode', 'r')
mutated_seq = np.array(json.loads(mutated_seq_file.read()))

eval_mutated_file = open(sys.argv[1] + 'EvalMutatedCode', 'r')
eval_mutated = np.array(json.loads(eval_mutated_file.read()))

nonclone_file = open(sys.argv[1] + 'indiciesNonClone', 'r')
nonclone_seq = np.array(json.loads(nonclone_file.read()))

eval_nonclone_file = open(sys.argv[1] + 'EvalMutatedCode', 'r')
eval_nonclone = np.array(json.loads(eval_nonclone_file.read()))
# origin_encoder_states = model.get_encoder_status(np.append(orig_seq, orig_seq[:nonclone_seq.shape[0]]))
# mutated_encoder_states = np.append(mutated_seq, nonclone_seq)
# answ = np.append(np.zeros(orig_seq.shape[0]), np.ones(nonclone_seq.shape[0]), axis=0)

origin_encoder_states = model.get_encoder_status(orig_seq[:30000])
mutated_encoder_states = model.get_encoder_status(np.append(mutated_seq[:20000], nonclone_seq[:10000]))
answ = np.append(np.zeros(20000), np.ones(10000), axis=0)

# eval_orig_encoder_states = model.get_encoder_status(np.append(eval_seq, eval_seq[:eval_nonclone.shape[0]], axis=0))
# eval_clone_encoder_states = model.get_encoder_status(np.append(eval_mutated, eval_nonclone, axis=0))

eval_orig_encoder_states = model.get_encoder_status(eval_seq)
eval_clone_encoder_states = model.get_encoder_status(eval_mutated)


print(len(origin_encoder_states))
print(len(mutated_encoder_states))
print(len(answ))

# LSTM RNN model
# _________________

lstm_model = lstm_siamese.LSTM(origin_encoder_states[0].shape[1], batch_size, layers)

loss_summary = tf.summary.scalar("loss", lstm_model.loss)
acc_summary = tf.summary.scalar("accuracy", lstm_model.accuracy)

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
    _, loss, dist, temp_sim = \
        sess.run([lstm_model.train_op, lstm_model.loss, lstm_model.distance, lstm_model.temp_sim],  feed_dict)
    print("TRAIN: step {}, loss {:g}".format(step, loss))
    print(y_batch, dist, temp_sim)


data = np.asarray(list(zip(origin_encoder_states, mutated_encoder_states, answ)))
data_size = data.shape[0]
batch_inds = np.random.permutation(data_size)
batches = data[batch_inds]
#batches = (origin_encoder_states, mutated_encoder_states, answ)
ptr = 0
max_validation_acc = 0.0

print(data_size)
for nn in range(data_size):
    x1_batch = []
    x2_batch = []
    y_batch = batches[nn][2]

    # size_diff = abs(len(batches[0][nn]) - len(batches[1][nn]))
    size_diff = abs(batches[nn][0].shape[0] - batches[nn][1].shape[0])

    if batches[nn][0].shape[0] < batches[nn][1].shape[0]:
        x1_batch = np.append(batches[nn][0], np.zeros((size_diff, batches[nn][0].shape[1])), axis=0)
        x2_batch = batches[nn][1]
    else:
        x1_batch = batches[nn][0]
        x2_batch = np.append(batches[nn][1], np.zeros((size_diff, batches[nn][1].shape[1])), axis=0)

    train_step(x1_batch, x2_batch, y_batch, nn)
    # current_step = tf.train.global_step(sess, global_step)
    # sum_acc = 0.0

eval_data = np.asarray(list(zip(eval_orig_encoder_states, eval_clone_encoder_states)))
data_size = data.shape[0]
batch_inds = np.random.permutation(data_size)
eval_batches = data[batch_inds]
for nn in range(eval_batches.shape[0]):
    x1_batch = []
    x2_batch = []
    size_diff = abs(eval_batches[nn][0].shape[0] - len(eval_batches[nn][1]))

    if eval_batches[nn][0].shape[0] < eval_batches[nn][1].shape[0]:
        x1_batch = np.append(eval_batches[nn][0], np.zeros((size_diff, eval_batches[nn][0].shape[1])), axis=0)
        x2_batch = eval_batches[nn][1]
    else:
        x1_batch = eval_batches[nn][0]
        x2_batch = np.append(eval_batches[nn][1], np.zeros((size_diff, eval_batches[nn][1].shape[1])), axis=0)

    feed_dict = {
            lstm_model.input_x1: x1_batch,
            lstm_model.input_x2: x2_batch,
        }
    dist, sim = sess.run([lstm_model.distance, lstm_model.temp_sim], feed_dict)
    print("EVAL: step {}".format(nn))
    print(dist, sim)

