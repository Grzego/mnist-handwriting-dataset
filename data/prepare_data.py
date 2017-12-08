import pickle
import numpy as np

# 1) go to page: https://edwin-de-jong.github.io/blog/mnist-sequence-data
# 2) download digit-images-thinned.tar.gz
# 3) unpack it in this directory (you should end up with: data/digit-images-thinned)
# 4) run: python prepare_data.py


def main():
    path = 'digit-images-thinned/trainimg-{}-thinned.txt'
    data_size = 60000
    data = np.empty((data_size, 28, 28, 1), dtype=np.uint8)
    for i in range(data_size):
        print('\r[{:5d}/{:5d}]'.format(i + 1, data_size), end='')
        with open(path.format(i), 'r') as file:
            img = [[int(x) for x in line.split()] for line in file]
            data[i, ...] = np.array(img, dtype=np.uint8).reshape((28, 28, 1))
    targets = np.array([int(x) for x in open('digit-images-thinned/trainlabels.txt', 'r')], dtype=np.int32)
    print('\nTraining set done.')

    path = 'digit-images-thinned/testimg-{}-thinned.txt'
    test_size = 10000
    test_data = np.empty((test_size, 28, 28, 1), dtype=np.uint8)
    for i in range(test_size):
        print('\r[{:5d}/{:5d}]'.format(i + 1, test_size), end='')
        with open(path.format(i), 'r') as file:
            img = [[int(x) for x in line.split()] for line in file]
            test_data[i, ...] = np.array(img, dtype=np.uint8).reshape((28, 28, 1))
    test_targets = np.array([int(x) for x in open('digit-images-thinned/testlabels.txt', 'r')], dtype=np.int32)
    print('\nTest set done.')

    dataset = {'data': data, 'targets': targets,
               'test_data': test_data, 'test_targets': test_targets}
    pickle.dump(dataset, open('mnist-thinned.pkl', 'wb'))


if __name__ == '__main__':
    main()
