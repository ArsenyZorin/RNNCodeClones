import numpy as np
import tensorflow as tf
import os
import glob
import re


def batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)

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
    files = glob.glob(directory + '/*.meta*')
    if len(files) > 0:
        try:
            file = re.sub('-*.meta', '', files[len(files) - 1])
            saver.restore(sess, file)
        except Exception as e:
            print('Serialization load error {}'.format(e))
            return False, None
        return True, sess
    return False, None
