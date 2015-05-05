from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD


def get_model():
    model = Sequential()

    init = 'he_normal'
    activation = 'sigmoid'

    model.add(Dense(3072, 1024, init=init, activation=activation))
    model.add(Dropout(0.1))

    model.add(Dense(1024, 1024, init=init, activation=activation))
    model.add(Dropout(0.1))

    model.add(Dense(1024, 256, init=init, activation=activation))
    model.add(Dropout(0.1))

    model.add(Dense(256, 64, init=init, activation=activation))
    model.add(Dropout(0.1))

    model.add(Dense(64, 10, init=init, activation='softmax'))

    return model


def get_optimizer():
    optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    return optimizer


def get_compiled_model():
    model = get_model()
    sgd = get_optimizer()
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def train_model(model, test_data, test_labels, train_data, train_labels):
    batch_size = 1024
    model.evaluate(test_data, test_labels, batch_size=batch_size, show_accuracy=True)
    model.fit(train_data, train_labels, nb_epoch=50, batch_size=batch_size, show_accuracy=True)