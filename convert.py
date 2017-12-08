import os
import cv2
import argparse
import h5py as h5
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import pickle
import seaborn
from matplotlib.animation import FuncAnimation
from sklearn.datasets import fetch_mldata
from collections import namedtuple

# ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=None)
parser.add_argument('--dataset', dest='dataset_path', type=str, default='data/mnist-thinned.pkl')
parser.add_argument('--save', dest='save', type=str)
parser.add_argument('--cpu', dest='cpu', action='store_true', default=False)
args = parser.parse_args()
# ---


def main():
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    config = config if args.cpu else None

    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(args.model_path + '.meta')
        saver.restore(sess, args.model_path)

        fields = ['handwriting', 'input', 'label', 'generated']
        vs = namedtuple('Params', fields)(
            *[tf.get_collection(name)[0] for name in fields]
        )

        mnist = pickle.load(open(args.dataset_path, 'rb'))
        data = np.concatenate([mnist['data'], mnist['test_data']], axis=0).astype(np.float32)
        targets = np.eye(10, dtype=np.float32)[
            np.concatenate([mnist['targets'], mnist['test_targets']], axis=0)]

        batch_size = 128
        lines = np.empty((data.shape[0], 20, 3), dtype=np.float32)
        for i in range(0, data.shape[0], batch_size):
            lines[i: i + batch_size], generated = sess.run([vs.handwriting, vs.generated],
                                                           feed_dict={vs.input: data[i: i + batch_size].reshape(
                                                               (-1, 28, 28, 1)),
                                                               vs.label: targets[i: i + batch_size]})
            print('\r[{:5d}/{:5d}]'.format(min(data.shape[0], i + batch_size), data.shape[0]), end='')

        lines[:, :, 2] = np.log(lines[:, :, 2]) <= 4.

        dataset = {'lines': lines,
                   'labels': np.argmax(targets, axis=-1).astype(np.int32)}
        pickle.dump(dataset, open('{}.pkl'.format(args.save), 'wb'))


if __name__ == '__main__':
    main()
