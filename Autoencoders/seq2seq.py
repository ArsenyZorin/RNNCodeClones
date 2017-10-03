import numpy as np
import tensorflow as tf
import json
from model import Seq2seq

tf.reset_default_graph()
sess = tf.InteractiveSession()

print(tf.__version__)

weights_file = open('weights', 'r')
weights = np.array(json.loads(weights_file.read()))

directory = 'trainedModel'

vocab_size = weights.shape[0]
vocab_lower = 2
vocab_upper = vocab_size

length_from = 1
length_to = 8

batch_size = 100
max_batches = 5000
batches_in_epoch = 1000

input_embedding_size = weights.shape[1]

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units


'''Try model.py'''
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

model = Seq2seq(encoder_cell, decoder_cell, vocab_size, input_embedding_size, weights)
model.train(length_from, length_to, vocab_lower, vocab_size,
     batch_size, max_batches, batches_in_epoch, directory)

origin_seq_file = open('indiciesOriginCode', 'r')
orig_seq = np.array(json.loads(origin_seq_file.read()))

mutated_seq_file = open('indiciesMutatedCode', 'r')
mutated_seq = np.array(json.loads(mutated_seq_file.read()))

origin_encoder_states = model.get_encoder_status(orig_seq)
mutated_encoder_states = model.get_encoder_status(mutated_seq)
