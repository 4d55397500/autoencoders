import numpy as np
import tensorflow as tf



class SemanticHashing(object):

    def __init__(self, xdim, hdim):

        self.xdim = xdim
        self.hdim = hdim

        w_encode = tf.Variable(tf.random_normal([xdim, hdim]))
        b_encode = tf.Variable(tf.random_normal([hdim]))

        w_decode = tf.Variable(tf.random_normal([hdim, xdim]))
        b_decode = tf.Variable(tf.random_normal([xdim]))

        self.noise = tf.placeholder('float', shape=[None, hdim])
        self.x_in = tf.placeholder('float', [None, xdim])
        self.h = tf.nn.sigmoid(tf.add(tf.matmul(self.x_in, w_encode), b_encode) + self.noise)
        self.output = tf.nn.sigmoid(tf.add(tf.matmul(self.h, w_decode), b_decode))
        self.x_out = tf.placeholder('float', [None, xdim])
        self.loss = tf.reduce_mean(tf.square(self.x_out - self.output))
        self.optimizer = tf.train.AdamOptimizer(0.1).minimize(self.loss)
        self.initializer = tf.global_variables_initializer()
        self.session = tf.Session()

    def train(self, x_train, batch_size, n_epochs):

        self.session.run(self.initializer)
        noise_std = 1.0
        for epoch in range(n_epochs):
            epoch_loss = 0.
            noise_std *= 1.05
            noise = np.random.normal(scale=noise_std, size=(batch_size, self.hdim))
            for i in range(int(x_train.shape[0] / batch_size)):
                batch = x_train[i * batch_size: (i + 1) * batch_size]
                _, batch_loss = self.session.run([self.optimizer, self.loss],
                                                 feed_dict={self.x_in: batch, self.x_out: batch, self.noise: noise})
                epoch_loss += batch_loss
            print(f"Epoch: {epoch} Epoch loss: {epoch_loss} Noise std: {noise_std}")

    def encode(self, x):
        return self.session.run(self.h, feed_dict={self.x_in: x, self.noise: np.random.normal(0.0, 0.0, size=(x.shape[0], self.hdim))})

    def decode(self, x):
        return self.session.run(self.output, feed_dict={self.x_in: x})

