import numpy as np
import tensorflow as tf
import json
import tensorflow.contrib.legacy_seq2seq as seq2seq
from tensorflow.contrib.rnn import GRUCell
import tempfile
logdir = tempfile.mkdtemp()

file = open("embeddingOriginCode", "r")
vector = json.loads(file.read())

batch = np.array(vector[3])
print(type(batch)) # numpy.ndarray
print(batch.shape) # (vocab_size, embedding_dim)

tf.reset_default_graph()
sess = tf.InteractiveSession()

seq_length = len(vector[3])
batch_size = 100
vocab_size = 100
embedding_dim = 100
memory_dim = 100

enc_inp = [tf.placeholder(tf.float32, shape=batch.shape, name="inp%i" % t) for t in range(seq_length)]
labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" % t) for t in range(seq_length)]
weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))

cell = GRUCell(memory_dim)
dec_outputs, dec_memory = seq2seq.basic_rnn_seq2seq(enc_inp, dec_inp, cell, dtype=tf.float32)

loss = seq2seq.sequence_loss(dec_outputs, labels, weights)
tf.summary.scalar("loss", loss)

magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
tf.summary.scalar("magnitude at t=1", magnitude)

summary_op = tf.summary.merge_all()

learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)

logdir = tempfile.mkdtemp()
summary_writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())


def train_batch(batch_size):
    X = batch
    #X = [np.random.choice(batch[0], size=batch.shape, replace=True) for _ in range(batch_size)]
    Y = X[:]

    # Dimshuffle to seq_len * batch_size
    #X = np.array(X).T
    #Y = np.array(Y).T

    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary

print("After train_batch definition")
for t in range(5000):
    loss_t, summary = train_batch(batch_size)
    summary_writer.add_summary(summary, t)
summary_writer.flush()

X_batch = batch


'''X_batch = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
           for _ in range(10)]'''
X_batch = np.array(X_batch).T

feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
dec_outputs_batch = sess.run(dec_outputs, feed_dict)

print(X_batch)

print([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch])


