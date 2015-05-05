from keras.utils import np_utils
import numpy as np


def get_labels_number(batches):
    size = batches[0]['data'].shape[1]
    return size


def get_empty_batch(size):
    merged_batch = {'data': np.array([]).reshape(0, size), 'filenames': [], 'labels': []}
    return merged_batch


def append_batch(merged_batch, batch):
    merged_batch['data'] = np.concatenate((merged_batch['data'], batch['data']))
    merged_batch['filenames'] += batch['filenames']
    merged_batch['labels'] += batch['labels']


def merge_batches(batches):
    size = get_labels_number(batches)
    merged_batch = get_empty_batch(size)

    for batch in batches:
        append_batch(merged_batch, batch)

    return merged_batch


def to_categorical(batch, classes_number):
    batch['labels'] = np_utils.to_categorical(batch['labels'], classes_number)


def categorize(merged_train_batch, test_batch):
    to_categorical(merged_train_batch, classes_number=10)
    to_categorical(test_batch, classes_number=10)