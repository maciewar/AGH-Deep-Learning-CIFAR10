from model import get_compiled_model, train_model


def get_data_and_labels(test_batch):
    test_data = test_batch['data']
    test_labels = test_batch['labels']
    return test_data, test_labels


def get_trained_model(train_batches, test_batch, weights_in=None, weights_out=None):
    model = get_compiled_model()

    test_data, test_labels = get_data_and_labels(test_batch)
    train_data, train_labels = get_data_and_labels(train_batches)

    if weights_in:
        model.load_weights(weights_in)

    train_model(model, test_data, test_labels, train_data, train_labels)

    if weights_out:
        model.save_weights(weights_out)

    return model


def get_predictions(model, batch):
    data = batch['data']
    filenames = batch['filenames']
    labels = model.predict_classes(data)

    # label_names = data_loader.load_label_names()
    predictions = [(filenames[i], label) for i, label in enumerate(labels)]
    return predictions