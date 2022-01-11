"""
IMPORTANT !!!

Require scipy-1.4.0.dev0+566c09b downloadable from https://github.com/indra-ipd/scipy

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import collections
import time
from sklearn.model_selection import train_test_split

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
Xtrain, Ytrain, Xtest, Ytest = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


seed = 1000
print(seed)
tf.set_random_seed(seed)
image_size = 28
chan_num = 1
labels_size = 10
learning_rate = 0.01
batch_size = 512
batches = int(len(Ytrain) / batch_size)
epoch = 10
iterations = batches

training_data = tf.placeholder('float', [None, image_size * image_size * chan_num])
labels = tf.placeholder('float')

algo = ['oSR1N', 'oSR1', 'oLBFGS', 'Adam', 'oLNAQ']  # 'oMoSR1',
col = {'oLNAQ': 'b', 'oMoSR1': 'k', 'oSR1': 'orange', 'oSR1N': 'm', 'oLBFGS': 'g', 'Adam': 'r'}


def get_batches(x_tr, y_tr, size):
    num_batch = int(len(y_tr) / size)
    data = []
    lab = []
    for i in range(num_batch):
        data.append(x_tr[i * size:i * size + size])
        lab.append(y_tr[i * size:i * size + size])
    return data, lab


Xtr, Ytr = get_batches(Xtrain, Ytrain, batch_size)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(X):
    # Here we defind the CNN architecture (LeNet-5)

    # Reshape input to 4-D vector
    input_layer = tf.reshape(X, [-1, 28, 28, 1])  # -1 adds minibatch support.

    # Padding the input to make it 32x32. Specification of LeNET
    padded_input = tf.pad(input_layer, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

    # Convolutional Layer #1
    # Has a default stride of 1
    # Output: 28 * 28 * 6
    conv1 = tf.layers.conv2d(
        inputs=padded_input,
        filters=6,  # Number of filters.
        kernel_size=5,  # Size of each filter is 5x5.
        padding="valid",  # No padding is applied to the input.
        activation=tf.nn.relu)

    # Pooling Layer #1
    # Sampling half the output of previous layer
    # Output: 14 * 14 * 6
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Output: 10 * 10 * 16
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,  # Number of filters
        kernel_size=5,  # Size of each filter is 5x5
        padding="valid",  # No padding
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Output: 5 * 5 * 16
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Reshaping output into a single dimention array for input to fully connected layer
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

    # Fully connected layer #1: Has 120 neurons
    dense1 = tf.layers.dense(inputs=pool2_flat, units=120, activation=tf.nn.relu)

    # Fully connected layer #2: Has 84 neurons
    dense2 = tf.layers.dense(inputs=dense1, units=84, activation=tf.nn.relu)

    # Output layer, 10 neurons for each digit
    logits = tf.layers.dense(inputs=dense2, units=10)

    return logits


perm_idx = []
for num in range(epoch):
    np.random.seed(100)
    perm_idx.append(np.random.permutation(len(Ytrain)))


def get_batches(x_tr, y_tr, size, ep_num):
    # shuffle data
    # idx = np.random.permutation(len(y_tr))
    idx = perm_idx[ep_num - 1]
    x_tr, y_tr = x_tr[idx], y_tr[idx]

    num_batch = int(len(y_tr) / size)
    data = []
    lab = []
    for i in range(num_batch):
        data.append(x_tr[i * size:i * size + size])
        lab.append(y_tr[i * size:i * size + size])
    return data, lab


def update(l, a):
    global train_loss, train_acc
    train_loss = l
    train_acc = a


count = 0
for meth in algo:

    tf.reset_default_graph()
    tf.set_random_seed(seed)

    training_data = tf.placeholder('float', [None, image_size * image_size * chan_num])
    labels = tf.placeholder('float')

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    output = convolutional_neural_network(training_data)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output))

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    """perm_idx = []
    for num in range(epoch):
        perm_idx.append(np.random.permutation(len(Ytrain)))"""

    color = col[meth]
    if meth == 'Adam':
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
        # epoch = 20
        timePlt = collections.deque(maxlen=iterations * epoch)
        errPlt = collections.deque(maxlen=iterations * epoch)

    else:
        m = 8
        # mu = 0.8
        sk_vec = collections.deque(maxlen=m)
        yk_vec = collections.deque(maxlen=m)
        alpha_k = collections.deque(maxlen=1)
        mu_val = collections.deque(maxlen=1)
        timePlt = collections.deque(maxlen=iterations * epoch)
        errPlt = collections.deque(maxlen=iterations * epoch)
        muHist = collections.deque(maxlen=iterations * epoch)

        alpha_k.append(1)
        vk_vec = collections.deque(maxlen=1)
        vk_vec.append(0)
        dirNorm = True

        if meth == 'oLNAQ' or meth == 'oLBFGS':  # vk_vec=None, sk_vec=None, yk_vec=None, m=8, alpha_k=1.0, mu=None, dirNorm=True,
            delta = alpha_k
            train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method=meth.lower(),
                options={'maxiter': iterations, 'disp': False, 'vk_vec': vk_vec, 'sk_vec': sk_vec, 'yk_vec': yk_vec,
                         'timeplot': timePlt, 'err': errPlt,
                         'm': m, 'alpha_k': alpha_k, 'muk': mu_val, 'dirNorm': dirNorm})

        elif meth == 'oLMoQ':  # vk_vec=None, sk_vec=None, yk_vec=None, m=8, alpha_k=1.0, mu=None, dirNorm=True,
            # grad_curr = collections.deque(maxlen=1)
            # grad_pre = collections.deque(maxlen=1)
            gfk_vec = collections.deque(maxlen=2)
            delta = alpha_k
            train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method=meth.lower(),
                options={'maxiter': iterations, 'disp': False, 'vk_vec': vk_vec, 'sk_vec': sk_vec, 'yk_vec': yk_vec,
                         # 'grad_pre': grad_pre, 'grad_curr': grad_curr,
                         'timeplot': timePlt, 'err': errPlt, 'gfk_vec': gfk_vec,
                         'm': m, 'alpha_k': alpha_k, 'muk': mu_val, 'dirNorm': dirNorm})

        elif meth == 'oSR1N':  # vk_vec=None, sk_vec=None, yk_vec=None, m=8, alpha_k=1.0, mu=None, dirNorm=True,
            # grad_curr = collections.deque(maxlen=1)
            # grad_pre = collections.deque(maxlen=1)
            gfk_vec = collections.deque(maxlen=2)
            theta = collections.deque(maxlen=1)
            delta = collections.deque(maxlen=1)
            theta.append(1)
            delta.append(1)

            train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method=meth.lower(),
                options={'maxiter': iterations, 'disp': False, 'vk_vec': vk_vec, 's_vec': sk_vec, 'y_vec': yk_vec,
                         'timeplot': timePlt, 'errHistory': errPlt, 'gfk_vec': gfk_vec, 'muHist': muHist,
                         'm': m, 'alpha_k': alpha_k, 'thetak': theta, 'delta_k': delta, 'dirNorm': dirNorm})

        elif meth == 'oMoSR1':
            gfk_vec = collections.deque(maxlen=2)
            theta = collections.deque(maxlen=1)
            delta = collections.deque(maxlen=1)
            theta.append(1)
            delta.append(1)

            train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method=meth.lower(),
                options={'maxiter': iterations, 'disp': False, 'vk_vec': vk_vec, 's_vec': sk_vec, 'y_vec': yk_vec,
                         'timeplot': timePlt, 'errHistory': errPlt, 'gfk_vec': gfk_vec, 'muHist': muHist,
                         'm': m, 'alpha_k': alpha_k, 'thetak': theta, 'delta_k': delta, 'dirNorm': dirNorm})

        elif meth == 'oSR1':
            gfk_vec = collections.deque(maxlen=2)
            theta = collections.deque(maxlen=1)
            delta = collections.deque(maxlen=1)
            theta.append(1)
            delta.append(1)

            train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method=meth.lower(),
                options={'maxiter': iterations, 'disp': False, 'vk_vec': vk_vec, 's_vec': sk_vec, 'y_vec': yk_vec,
                         'timeplot': timePlt, 'errHistory': errPlt, 'gfk_vec': gfk_vec,
                         'm': m, 'alpha_k': alpha_k, 'thetak': theta, 'delta_k': delta, 'dirNorm': dirNorm})

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print("Initial Error of ", meth, " : ", sess.run(loss, feed_dict={training_data: Xtrain, labels: Ytrain}))

    step = 0
    alpha_k.append(1)
    eta0 = batch_size / (batch_size + 2)

    test_acc_plot = []
    test_loss_plot = []
    train_acc_plot = []
    train_loss_plot = []

    for ep in range(1, epoch + 1):
        print("EPOCH ", ep, " : ##########")
        Xtr, Ytr = get_batches(Xtrain, Ytrain, batch_size, ep - 1)
        theta_k = 1
        alpha_k.append(alpha_k[-1] * 0.5)

        for i in range(iterations):
            step += 1

            data, lab = Xtr[i], Ytr[i]
            feed_dict = {training_data: data, labels: lab}

            if meth == 'Adam':
                start = time.time()
                _, train_loss, train_acc = sess.run([train_step, loss, accuracy], feed_dict=feed_dict)
                end = time.time()
                timePlt.append(end - start)

            else:
                if meth != 'oLMoQ':
                    if step > 1:
                        alpha_k.append(1 / np.sqrt(step))
                    else:
                        alpha_k.append(0.5)

                if meth == 'oLMoQ':
                    if step < 1:
                        alpha_k.append(0.5)

                if meth == 'oLBFGS':
                    mu_val.append(0)
                    mu = mu_val[-1]

                if meth == 'oLNAQ':
                    mu = 0.85
                    mu_val.append(0.85)
                    mu = mu_val[-1]

                if meth == 'oLMoQ':
                    # theta_kp1 = ((1e-5 - (theta_k * theta_k)) + np.sqrt(((1e-5 - (theta_k * theta_k)) * (1e-5 - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
                    # mu = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1), 0.95)
                    # theta_k = theta_kp1

                    mu_val.append(0.85)
                    mu = mu_val[-1]

                res = train_step.minimize(sess, fetches=[loss, accuracy],
                                          loss_callback=update,
                                          feed_dict=feed_dict)

            test_loss, test_acc = sess.run([loss, accuracy], feed_dict={training_data: Xtest, labels: Ytest})
            test_acc_plot.append(test_acc * 100)
            test_loss_plot.append(test_loss)
            train_acc_plot.append(train_acc * 100)
            train_loss_plot.append(train_loss)

            try:
                if i % 50 == 0:
                    print(
                        'Step {}; train loss {}; train accuracy {}; test loss {}; test accuracy {}; alpha {}; mu {}'.format(
                            i, train_loss, train_acc * 100, test_loss, test_acc * 100, delta[-1], muHist[-1]))

            except:
                if i % 50 == 0:
                    print(
                        'Step {}; train loss {}; train accuracy {}; test loss {}; test accuracy {}; alpha {}'.format(
                            i, train_loss, train_acc * 100, test_loss, test_acc * 100, delta[-1]))
    #print(muHist)

    leg = algo


    timePlt.clear()
    sk_vec.clear()
    yk_vec.clear()
    vk_vec.clear()
    gfk_vec.clear()

    sess.close()

print('seed: ', seed)

