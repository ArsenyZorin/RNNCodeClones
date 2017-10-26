import numpy as np
import tensorflow as tf
import os
from itertools import zip_longest


def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_time_major, sequence_lengths


def siam_batches(x1, x2, x3=None):
    if x3 is not None:
        data = np.asarray(list(zip(x1, x2, x3)))
    else:
        data = np.array(list(zip(x1, x2)))

    data_size = data.shape[0]
    batch_inds = np.random.permutation(data_size)
    return data[batch_inds]


def shape_diff(x1, x2):
    size_diff = abs(x1.shape[0] - x2.shape[0])

    if x1.shape[0] < x2.shape[0]:
        x1_batch = np.append(x1, np.zeros((size_diff, x1.shape[1])), axis=0)
        x2_batch = x2
    else:
        x1_batch = x1
        x2_batch = np.append(x2, np.zeros((size_diff, x2.shape[1])), axis=0)
    return x1_batch, x2_batch


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
        raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]


def save_model(directory, name, sess):
    if os.path.exists(directory):
        os.rmdir(directory)

    builder = tf.saved_model.builder.SavedModelBuilder(directory)
    builder.add_meta_graph_and_variables(sess, [name])
    builder.save()
    print('Exporting train model to {}'.format(directory))


def load_model(saver, sess, directory):
    if os.path.exists(directory + '.meta'):
        try:
            saver.restore(sess, directory)
        except Exception:
            print('Serialization load error')
            return False, None
        return True, sess
    return False, None
