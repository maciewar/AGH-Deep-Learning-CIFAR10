import cProfile
import data_loader
import data_manipulator
import data_saver
import neural_net


def main():
    test_batch, train_batch = data_loader.load_data()
    data_manipulator.categorize(train_batch, test_batch)

    model = neural_net.get_trained_model(train_batches=train_batch,
                                         test_batch=test_batch,
                                         weights_in='weights/1024_1024_256_64_epochs_45',
                                         weights_out='weights/1024_1024_256_64_epochs_50')

    predictions = neural_net.get_predictions(model, test_batch)
    data_saver.save_results("results/result.csv", predictions)


def profiling():
    cProfile.run('main()', sort='tottime')


if __name__ == "__main__":
    main()
