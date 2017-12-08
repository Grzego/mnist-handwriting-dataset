import os
import random
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# -----
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', type=str, default='data/mnist-handwriting.pkl')
args = parser.parse_args()
# -----


def main():
    # load datatset
    dataset = pickle.load(open(args.dataset, 'rb'))

    plt.ion()
    fig = plt.figure(figsize=(5, 5))
    fig.canvas.mpl_connect('close_event', lambda e: exit(0))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1)

    # loop over dataset randomly
    while True:
        r = random.randint(0, len(dataset['lines']) - 1)
        lines, label = dataset['lines'][r], dataset['labels'][r]

        ax.cla()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Digit {}'.format(label))

        for p1, p2 in zip(lines, lines[1:]):
            # p1[2] == 1 means the pen was lifted on that point
            # so we don't draw line from p1 to p2
            if p1[2] < 0.5:
                # coords are stored with point (0, 0) being in upper left corner
                # and in rendering (0, 0) is in bottom left corner, so we need to flip it
                ax.plot([p1[0], p2[0]], [1 - p1[1], 1 - p2[1]])
            ax.axis([0, 1, 0, 1])
            plt.draw()
            plt.pause(1. / 60.)
        plt.pause(1.)
        plt.cla()


if __name__ == '__main__':
    main()
