import numpy as np
import tensorflow as tf
import json
import sys
import os
from seq2seq_model import Seq2seq, SiameseNetwork

if len(sys.argv) < 2:
    print('Invalid usage of Seq2seq script')
    print('Please set directory with data')
    sys.exit(0)

if sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('Help message')
    sys.exit(0)

tf.flags.DEFINE_string('type', 'full', 'Type of evaluation. Could be: \n\ttrain\n\teval\n\tfull')
tf.flags.DEFINE_string('data', os.path.expanduser('~/.rnncodeclones'), 'Directory with data for analysis')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print('Arguments: {}'.format(sys.argv[1]))

tf.reset_default_graph()
print(tf.__version__)

directory_seq2seq = FLAGS.data + '/networks/trainedModel'
directory_lstm = FLAGS.data + '/networks/lstmTrainedModel'


def train():
    seq2seq_model.train(length_from, length_to, vocab_lower, vocab_size,
                        batch_size, max_batches, batches_in_epoch, directory_seq2seq)

    origin_seq_file = open(FLAGS.data + '/indiciesOriginCode', 'r')
    orig_seq = np.array(json.loads(origin_seq_file.read()))

    eval_seq_file = open(FLAGS.data + '/EvalCode', 'r')
    eval_seq = np.array(json.loads(eval_seq_file.read()))

    mutated_seq_file = open(FLAGS.data + '/indiciesMutatedCode', 'r')
    mutated_seq = np.array(json.loads(mutated_seq_file.read()))

    eval_mutated_file = open(FLAGS.data + '/EvalMutatedCode', 'r')
    eval_mutated = np.array(json.loads(eval_mutated_file.read()))

    nonclone_file = open(FLAGS.data + '/indiciesNonClone', 'r')
    nonclone_seq = np.array(json.loads(nonclone_file.read()))

    eval_nonclone_file = open(FLAGS.data + '/EvalNonClone', 'r')
    eval_nonclone = np.array(json.loads(eval_nonclone_file.read()))

    origin_encoder_states = seq2seq_model.get_encoder_status(np.append(orig_seq, orig_seq[:nonclone_seq.shape[0]]))
    mutated_encoder_states = seq2seq_model.get_encoder_status(np.append(mutated_seq, nonclone_seq))
    answ = np.append(np.zeros(orig_seq.shape[0]), np.ones(nonclone_seq.shape[0]), axis=0)

    # origin_encoder_states = model.get_encoder_status(orig_seq[:30000])
    # mutated_encoder_states = model.get_encoder_status(np.append(mutated_seq[:20000], nonclone_seq[:10000]))
    # answ = np.append(np.zeros(20000), np.ones(10000), axis=0)

    eval_orig_encoder_states = seq2seq_model.get_encoder_status(np.append(eval_seq, eval_seq[:eval_nonclone.shape[0]]))
    eval_clone_encoder_states = seq2seq_model.get_encoder_status(np.append(eval_mutated, eval_nonclone))
    eval_answ = np.append(np.zeros(eval_seq.shape[0]), np.ones(eval_nonclone.shape[0]))

    # LSTM RNN model
    # _________________

    lstm_model.train(origin_encoder_states, mutated_encoder_states, answ, directory_lstm)
    lstm_model.eval(eval_orig_encoder_states, eval_clone_encoder_states, eval_answ)


def eval():
    seq2seq_model.restore(directory_seq2seq + '/seq2seq.ckpt')
    origin_seq_file = open(FLAGS.data + '/indiciesOriginCode', 'r')
    orig_seq = np.array(json.loads(origin_seq_file.read()))
    encoder_states = seq2seq_model.get_encoder_status(orig_seq)
    lstm_model.eval(encoder_states, encoder_states)


weights_file = open(FLAGS.data + '/networks/pretrainedWeights', 'r')
weights = np.array(json.loads(weights_file.read()))

vocab_size = weights.shape[0]
vocab_lower = 2

length_from = 1
length_to = 1000

batch_size = 100
max_batches = 20000
batches_in_epoch = 1000

input_embedding_size = weights.shape[1]

layers = 5
encoder_hidden_units = layers
decoder_hidden_units = encoder_hidden_units

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

seq2seq_model = Seq2seq(encoder_cell, decoder_cell, vocab_size, input_embedding_size, weights)
lstm_model = SiameseNetwork(layers, batch_size, layers)

if 'train' == FLAGS.type:
    train()
elif 'eval' == FLAGS.type:
    eval()
if 'full' == FLAGS.type:
    train()
    eval()



