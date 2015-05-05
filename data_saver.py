import cPickle
import os


def create_path(filename):
    directory_name = os.path.dirname(filename)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def save_batch(filename, batch):
    create_path(filename)
    cPickle.dump(batch, open(filename, 'wb'))


def save_results(filename, results):
    create_path(filename)
    with open(filename, 'wb') as result_file:
        for result in results:
            image_file, label_name = result
            result_file.write('{0}, {1}\n'.format(image_file, label_name))