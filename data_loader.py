import cPickle
from data_manipulator import merge_batches


def unpickle(batch_file):
    with open(batch_file, 'rb') as file_handler:
        batch = cPickle.load(file_handler)
        return batch


def load_batches(batches_number):
    data_batches = [unpickle('data/data_batch_{0}'.format(i + 1)) for i in xrange(batches_number)]
    return data_batches


def load_test_batch():
    test_batch = unpickle('data/test_batch')
    return test_batch


def load_train_batch():
    train_batches = load_batches(5)
    merged_train_batch = merge_batches(train_batches)
    return merged_train_batch


def load_data():
    train_batch = load_train_batch()
    test_batch = load_test_batch()
    return test_batch, train_batch


def load_label_names():
    batches_meta = unpickle('data/batches.meta')
    label_names = batches_meta['label_names']
    return label_names