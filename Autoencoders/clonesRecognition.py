import numpy as np
import tensorflow as tf
import json
import sys
import os

import time
import shutil
from model import Seq2seq, SiameseNetwork

tf.flags.DEFINE_string('type', 'full', 'Type of evaluation. Could be: \n\ttrain\n\teval\n\tfull')
tf.flags.DEFINE_string('data', os.path.expanduser('~/.rnncodeclones'), 'Directory with data for analysis')
tf.flags.DEFINE_integer('cpus', 1, 'Amount of threads for evaluation')
tf.flags.DEFINE_integer('gpus', None, 'Amount of GPUs for training')

FLAGS = tf.flags.FLAGS


def train(cell, layers, length, vocab, weights, batch, seq2seq_dir, siam_dir, vectors_dir):
    model = {'seq2seq': seq2seq_train(cell, length, vocab, weights, batch, seq2seq_dir)}
    model['siam'] = siam_train(vectors_dir, model['seq2seq'], batch['size'], layers, siam_dir)
    return model


def eval(model, vectors_dir):
    file = open(vectors_dir + '/originCode', 'r')
    seq = np.array(json.loads(file.read()))
    states = model['seq2seq'].get_encoder_status(seq)

    return model['siam'].eval(states)


def seq2seq_train(cell, length, vocab, weights, batch, directory):
    seq2seq_model = Seq2seq(cell['encoder'], cell['decoder'], vocab['size'], weights.shape[1], weights)
    seq2seq_model.train(length, vocab, batch, directory)
    return seq2seq_model


def siam_train(vectors, seq2seq_model, batch_size, layers, directory):
    orig_file = open(vectors + '/indiciesOriginCode', 'r') # correct indices
    mutated_file = open(vectors + '/indiciesMutatedCode', 'r')
    nonclone_file = open(vectors + '/indiciesNonClone', 'r')

    orig_seq = np.array(json.loads(orig_file.read()))
    mutated_seq = np.array(json.loads(mutated_file.read()))
    nonclone_seq = np.array(json.loads(nonclone_file.read()))

    orig_encst = seq2seq_model.get_encoder_status(np.append(orig_seq, orig_seq[:nonclone_seq.shape[0]]))
    mut_encst = seq2seq_model.get_encoder_status(np.append(mutated_seq, nonclone_seq))
    answ = np.append(np.zeros(orig_seq.shape[0]), np.ones(nonclone_seq.shape[0]), axis=0)

    siam_model = SiameseNetwork(orig_encst[0].shape[1], batch_size, layers)
    siam_model.train(orig_encst, mut_encst, answ, directory)
    return siam_model


def restore_models(dirs, cell, vocab, length, weights, batch, layers):
    model = {'seq2seq': Seq2seq(cell['encoder'], cell['decoder'], vocab['size'], weights.shape[1], weights),
             'siam': SiameseNetwork(layers, batch['size'], layers)}

    if model['seq2seq'].restore(dirs['seq2seq']) is None:
        seq2seq_train(cell, length, vocab, weights, batch, dirs['seq2seq'])
    if model['siam'].restore(dirs['siam']) is None:
        siam_train(dirs['vecs'], model['seq2seq'], batch['size'], layers, dirs['siam'])
    return model


def show_time(start):
    end = time.time()
    secs = round(end - start, 3)
    mins = 0
    hour = 0

    if secs > 59:
        mins = (int)(secs / 60)
        secs -= mins * 60
        if mins > 59:
            hour = (int)(mins / 60)
            mins -= hour * 60

    if mins < 10:
        mins = '0' + str(mins)
    print('\nElapsed time: {}:{}:{}'.format(hour, mins, round(secs, 3)))


def main(_):
    start = time.time()

    try:
        tf.reset_default_graph()
        print(tf.__version__)

        if FLAGS.type != 'eval' and FLAGS.type != 'train' and FLAGS.type != 'full':
            print('Unknown type flag.')
            print('Allowable values are:')
            print('\ttrain\n\teval\n\tfull')
            show_time(start)
            sys.exit(1)

        dirs = {'seq2seq': FLAGS.data + '/networks/seq2seqModel',
               'siam': FLAGS.data + '/networks/siameseModel',
               'vecs': FLAGS.data + '/vectors'}

        if FLAGS.type != 'eval':
            if os.path.exists(dirs['seq2seq']):
                shutil.rmtree(dirs['seq2seq'])
            if os.path.exists(dirs['siam']):
                shutil.rmtree(dirs['siam'])
            os.mkdir(dirs['seq2seq'])
            os.mkdir(dirs['siam'])

        weights_file = open(FLAGS.data + '/networks/word2vec/pretrainedWeights', 'r')
        weights = np.array(json.loads(weights_file.read()))

        vocab = {'size': weights.shape[0], 'lower': 2}
        length = {'from': 1, 'to': 100}
        batch = {'size': 1000, 'max': 1500, 'epoch': 1000}

        layers = 5
        encoder_hidden_units = 10
        decoder_hidden_units = encoder_hidden_units

        if FLAGS.gpus is not None:
            enc_cells = []
            dec_cells = []
            for i in range(layers):
                enc_cells.append(tf.contrib.rnn.DeviceWrapper(
                    tf.contrib.rnn.LSTMCell(encoder_hidden_units),
                    '/gpu:%d' % (encoder_hidden_units % FLAGS.gpus)
                ))
                dec_cells.append(tf.contrib.rnn.DeviceWrapper(
                    tf.contrib.rnn.LSTMCell(decoder_hidden_units),
                    '/gpu:%d' % (decoder_hidden_units % FLAGS.gpus)
                ))
            cell = {'encoder': tf.contrib.rnn.MultiRNNCell(enc_cells),
                    'decoder': tf.contrib.rnn.MultiRNNCell(dec_cells)}
        else:
            cell = {'encoder': tf.contrib.rnn.LSTMCell(encoder_hidden_units),
                    'decoder': tf.contrib.rnn.LSTMCell(decoder_hidden_units)}

        if FLAGS.type == 'train':
            train(cell, layers, length, vocab, weights, batch, dirs['seq2seq'], dirs['siam'], dirs['vecs'])
        elif FLAGS.type == 'full':
            model = train(cell, layers, length, vocab, weights, batch, dirs['seq2seq'], dirs['siam'], dirs['vecs'])
            eval(model, dirs['vecs'])
        elif FLAGS.type == 'eval':
            model = restore_models(dirs, cell, vocab, length, weights, batch, layers)
            eval(model, dirs['vecs'])

        show_time(start)

    except KeyboardInterrupt:
        print('Keyboard interruption')
        show_time(start)
        sys.exit(0)


if __name__ == '__main__':
    tf.app.run()
