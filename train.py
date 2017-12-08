import os
import cv2
import random
import argparse
import h5py as h5
import numpy as np
import tensorflow as tf
import pickle
from collections import namedtuple

from utils import next_experiment_path

# -----
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
parser.add_argument('--epochs', dest='epochs', default=15, type=int)
parser.add_argument('--restore', dest='restore', default=None, type=str)
args = parser.parse_args()
# -----

epsilon = 1e-8


def draw_lines(lines):
    img = np.zeros((28, 28, 1), dtype=np.float32)
    for p1, p2 in zip(lines, lines[1:]):
        if p1[2] < 0.5:
            cv2.line(img, (int(28. * p1[0]), int(28. * p1[1])), (int(28. * p2[0]), int(28. * p2[1])),
                     (255, 255, 255), 1)
    return img.astype(np.float32) / 255.


def load_pretrain_data():
    with h5.File('data/samples.h5', 'r') as file:
        dataset = file['data'][:].astype(np.float32)
        labels = np.eye(10, dtype=np.float32)[file['labels'][:]]
    images = np.array([draw_lines(lines) for lines in dataset])
    dataset[:, :, 2] = 0.  # ignore end of line
    return images, dataset, labels


def create_graph(num_segments):
    graph = tf.Graph()
    with graph.as_default():
        def create_line_model(coords, images):
            def run_model(points, size=28):
                with tf.name_scope('line_model'):
                    def soften(x, soft):
                        return tf.sigmoid(x * soft)

                    # intersection line coordinates (vertical and horizontal)
                    xs = tf.expand_dims(tf.linspace(0.5 / size, 1. - 0.5 / size, size), axis=0)
                    # pixels intervals
                    pixels = tf.expand_dims(tf.expand_dims(tf.linspace(0., 1., size + 1), axis=1), axis=0)

                    mesh_y = tf.tile(pixels, multiples=[1, 1, size + 1])
                    mesh_x = tf.tile(tf.transpose(pixels, perm=[0, 2, 1]), multiples=[1, size + 1, 1])

                    softness = tf.expand_dims(points[:, 2:3], axis=2)
                    x1, y1, x2, y2 = points[:, 0:1], points[:, 1:2], points[:, 3:4], points[:, 4:5]

                    direction = points[:, 3:5] - points[:, 0:2]
                    direction /= tf.sqrt(tf.reduce_sum(tf.square(direction), axis=-1, keep_dims=True) + 1e-7)

                    inv_cos = 1. / (direction[:, 0:1] + 1e-8)
                    vertical = (xs - x1) * inv_cos * direction[:, 1:2] + y1
                    vertical = tf.expand_dims(vertical, axis=1)

                    inv_cos = 1. / (direction[:, 1:2] + 1e-8)
                    horizontal = (xs - y1) * inv_cos * direction[:, 0:1] + x1
                    horizontal = tf.expand_dims(horizontal, axis=1)

                    pixel_offset = 1. / size

                    lower_px = soften(vertical - pixels, softness)
                    upper_px = soften(pixels - vertical + pixel_offset, softness)
                    vcombined = lower_px * upper_px
                    vcombined = vcombined[:, :-1, :]

                    lower_px = soften(horizontal - pixels, softness)
                    upper_px = soften(pixels - horizontal + pixel_offset, softness)
                    hcombined = lower_px * upper_px
                    hcombined = hcombined[:, :-1, :]
                    hcombined = tf.transpose(hcombined, perm=[0, 2, 1])

                    on_line = tf.maximum(vcombined, hcombined)
                    xmin = tf.expand_dims(tf.minimum(x1, x2), axis=1)
                    xmax = tf.expand_dims(tf.maximum(x1, x2), axis=1)
                    ymin = tf.expand_dims(tf.minimum(y1, y2), axis=1)
                    ymax = tf.expand_dims(tf.maximum(y1, y2), axis=1)

                    rect_mask = soften(mesh_x - xmin, softness)
                    rect_mask *= soften(mesh_y - ymin, softness)
                    rect_mask *= soften(xmax - mesh_x + pixel_offset, softness)
                    rect_mask *= soften(ymax - mesh_y + pixel_offset, softness)

                    return tf.expand_dims(on_line * rect_mask[:, 1:, 1:], axis=3)

            return run_model, namedtuple('LineModel', ['coords', 'images'])(
                coords, images
            )

        def create_handwriting_model(images, targets, run_line_model):
            def run_handwriting_model(inputs, labels, reuse=True):
                with tf.variable_scope('handwriting', reuse=reuse):
                    batch_size = tf.shape(inputs)[0]
                    flatten = tf.reshape(inputs, [-1, np.prod(inputs.get_shape().as_list()[1:])])

                    cell1 = tf.nn.rnn_cell.LSTMCell(64 * 10)
                    cell2 = tf.nn.rnn_cell.LSTMCell(64 * 10)
                    cell3 = tf.nn.rnn_cell.LSTMCell(64 * 10, num_proj=3)

                    coord = tf.zeros(shape=[batch_size, 3])

                    state1 = cell1.zero_state(batch_size, dtype=tf.float32)
                    state2 = cell2.zero_state(batch_size, dtype=tf.float32)
                    state3 = cell3.zero_state(batch_size, dtype=tf.float32)

                    # this is rather computationally inefficient but is
                    # also really simple way to incorporate different models
                    digit_mask = tf.tile(labels, multiples=[1, 64])

                    # generate lines
                    coords = []
                    for s in range(num_segments):
                        with tf.variable_scope('points', reuse=reuse if s == 0 else True):
                            in1, state1 = cell1(tf.concat([flatten, labels, coord], axis=-1), state1,
                                                scope='cell_1')
                            in2, state2 = cell2(tf.concat([in1 * digit_mask, labels, coord], axis=-1), state2,
                                                scope='cell_2')
                            new_coord, state3 = cell3(tf.concat([in2 * digit_mask, labels, coord], axis=-1), state3,
                                                      scope='cell_3')
                            coord = tf.concat([tf.nn.sigmoid(new_coord[:, :2]),
                                               tf.exp(new_coord[:, 2:3])], axis=-1)
                            coords += [coord]

                # create image from lines
                generated = tf.zeros_like(inputs)
                for p1, p2 in zip(coords, coords[1:]):
                    generated += run_line_model(tf.concat([p1, p2], axis=-1))
                    # generated = tf.maximum(generated, run_line_model(tf.concat([p1, p2], axis=-1)))

                return tf.stack(coords, axis=1), generated

            handwriting, generated_images = run_handwriting_model(images, targets, reuse=None)

            tf.add_to_collection('handwriting', handwriting)
            tf.add_to_collection('input', images)
            tf.add_to_collection('label', targets)
            tf.add_to_collection('generated', generated_images)

            global_step = tf.get_variable('global_step', shape=[], dtype=tf.float32)

            with tf.variable_scope('handwriting/pretrain'):
                points = tf.placeholder(tf.float32, shape=[None, num_segments, 3])
                pretrain_loss = tf.losses.mean_squared_error(points[:, :, :2], handwriting[:, :, :2])
                pretrain = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(pretrain_loss)

            with tf.variable_scope('handwriting/train'):
                gen_loss = tf.losses.mean_squared_error(images, generated_images)
                # ---
                distances = tf.sqrt(tf.reduce_sum(tf.square(handwriting[:, 1:, :2] - handwriting[:, :-1, :2]),
                                                  axis=-1))
                dist_loss = tf.reduce_mean(distances)
                # ---
                vecs = handwriting[:, 1:, :2] - handwriting[:, :-1, :2]
                vecs /= tf.sqrt(tf.reduce_sum(tf.square(vecs), axis=-1, keep_dims=True) + 1e-7)
                angs = tf.reduce_sum(vecs[:, 1:, :] * vecs[:, :-1, :], axis=-1)
                ang_loss = 1. - tf.reduce_mean(angs)
                # ---
                loss = gen_loss
                loss += 0.01 * dist_loss
                loss += 0.01 * ang_loss
                vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='handwriting')
                train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, var_list=vars_to_train,
                                                                             global_step=global_step)

            with tf.name_scope('summary/handwriting'):
                summary = tf.summary.merge([
                    tf.summary.scalar('loss', loss),
                    tf.summary.scalar('gen_loss', gen_loss),
                    tf.summary.scalar('dist_loss', dist_loss),
                    tf.summary.scalar('ang_loss', ang_loss),
                    tf.summary.image('targets', images),
                    tf.summary.image('generated', generated_images),
                ])

            return namedtuple('HandwritingModel', ['train', 'loss', 'summary', 'images', 'labels', 'handwritting',
                                                   'points', 'pretrain_loss', 'pretrain'])(
                train, loss, summary, images, targets, handwriting,
                points, pretrain_loss, pretrain
            )

        # mnist or line targets
        images_ = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        coords_ = tf.placeholder(tf.float32, shape=[None, 6])
        labels_ = tf.placeholder(tf.float32, shape=[None, 10])

        run_line_model_, line_model = create_line_model(coords_, images_)
        handwriting_model = create_handwriting_model(images_, labels_, run_line_model_)

    return graph, line_model, handwriting_model


def main():
    restore_model = args.restore
    batch_size = args.batch_size
    num_epoch = args.epochs
    batches_per_epoch = 1000

    mnist = pickle.load(open('data/mnist-thinned.pkl', 'rb'))
    shuffle = np.random.permutation(len(mnist['data']))
    data = mnist['data'][shuffle].astype(np.float32) / 255.
    targets = np.eye(10, dtype=np.float32)[mnist['targets'][shuffle]]

    g, lg, hg = create_graph(num_segments=20)

    with tf.Session(graph=g) as sess:
        model_saver = tf.train.Saver(max_to_keep=2)
        if restore_model:
            model_file = tf.train.latest_checkpoint(os.path.join(restore_model, 'models'))
            experiment_path = restore_model
            epoch = int(model_file.split('-')[-1]) + 1
            model_saver.restore(sess, model_file)
        else:
            sess.run(tf.global_variables_initializer())
            experiment_path = next_experiment_path()
            epoch = 0
            print('Starting experiment: {}'.format(experiment_path))

        summary_writer = tf.summary.FileWriter(experiment_path, graph=g, flush_secs=10)
        summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START),
                                       global_step=epoch * batches_per_epoch)

        pretrain_iters = 100
        images, lines, labels = load_pretrain_data()
        print('\n\nPretraining handwriting...')
        for e in range(pretrain_iters):
            l, _ = sess.run([hg.pretrain_loss, hg.pretrain],
                            feed_dict={hg.images: images, hg.points: lines, hg.labels: labels})
            print('\r[{:5d}/{:5d}] loss = {}'.format(e + 1, pretrain_iters, l), end='')

        images, lines, labels = load_pretrain_data()
        print('\n\nTraining handwriting generation...')
        for e in range(epoch, num_epoch):
            print('\nEpoch {}'.format(e))
            for b in range(0, len(data), batch_size):
                # run iteration on collected samples
                pl, _ = sess.run([hg.pretrain_loss, hg.pretrain],
                                 feed_dict={hg.images: images, hg.points: lines, hg.labels: labels})
                # run iteration on MNIST samples
                batch_data = data[b: b + batch_size]
                batch_target = targets[b: b + batch_size]
                if b % (batch_size * 10) == 0:
                    l, s, _ = sess.run([hg.loss, hg.summary, hg.train],
                                       feed_dict={hg.images: batch_data, hg.labels: batch_target})
                    summary_writer.add_summary(s, global_step=e * len(data) + b)
                else:
                    l, _ = sess.run([hg.loss, hg.train],
                                    feed_dict={hg.images: batch_data, hg.labels: batch_target})
                print('\r[{:5d}/{:5d}] loss = {:16.10f}'.format(b + len(batch_data), len(data), l), end='')
                print('; pretrain = {:16.10f}'.format(pl), end='', flush=True)

            model_saver.save(sess, os.path.join(experiment_path, 'models', 'handwriting_model'),
                             global_step=e)


if __name__ == '__main__':
    main()
