import numpy as np
import tensorflow as tf
import json
import sys
from model import Seq2seq

if len(sys.argv) < 2:
    print('Invalid usage of Seq2seq script')
    print('Please set directory with data')
    sys.exit(0)

if sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('Help message')
    sys.exit(0)

print('Arguments: {}'.format(sys.argv[1]))

tf.reset_default_graph()
sess = tf.InteractiveSession()

print(tf.__version__)

weights_file = open(sys.argv[1] + 'pretrainedWeights', 'r')
weights = np.array(json.loads(weights_file.read()))

directory = 'trainedModel'

vocab_size = weights.shape[0]
vocab_lower = 2
vocab_upper = vocab_size

length_from = 5
length_to = 1000

batch_size = 100
max_batches = 1000
batches_in_epoch = 100

input_embedding_size = weights.shape[1]

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units


'''Try model.py'''
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

model = Seq2seq(encoder_cell, decoder_cell, vocab_size, input_embedding_size, weights)
model.train(length_from, length_to, vocab_lower, vocab_size,
     batch_size, max_batches, batches_in_epoch, directory)

origin_seq_file = open(sys.argv[1] + 'indiciesOriginCode', 'r')
orig_seq = np.array(json.loads(origin_seq_file.read()))

mutated_seq_file = open(sys.argv[1] + 'indiciesMutatedCode', 'r')
mutated_seq = np.array(json.loads(mutated_seq_file.read()))

origin_encoder_states = model.get_encoder_status(orig_seq)
mutated_encoder_states = model.get_encoder_status(mutated_seq)
