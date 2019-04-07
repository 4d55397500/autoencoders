
import numpy as np
import tensorflow as tf


class A1(object):

    def __init__(self, xdim, hdim):

        w_encode = tf.Variable(tf.random_normal([xdim, hdim]))
        b_encode = tf.Variable(tf.random_normal([hdim]))
        w_decode = tf.Variable(tf.random_normal([hdim, xdim]))
        b_decode = tf.Variable(tf.random_normal([xdim]))
        self.x_in = tf.placeholder('float', [None, xdim])
        self.h = tf.add(tf.matmul(self.x_in, w_encode), b_encode)
        self.output = tf.add(tf.matmul(self.h, w_decode), b_decode)
        self.x_out = tf.placeholder('float', [None, xdim])
        self.loss = tf.reduce_mean(tf.square(self.x_out - self.output))
        self.optimizer = tf.train.AdamOptimizer(0.05).minimize(self.loss)
        self.initializer = tf.global_variables_initializer()
        self.session = tf.Session()

    def train(self, x_train, batch_size, n_epochs):
        self.session.run(self.initializer)
        for epoch in range(n_epochs):
            epoch_loss = 0.
            for i in range(int(x_train.shape[0] / batch_size)):
                batch = x_train[i * batch_size: (i + 1) * batch_size]
                _, batch_loss = self.session.run([self.optimizer, self.loss],
                                                 feed_dict={self.x_in: batch, self.x_out: batch})
                epoch_loss += batch_loss
            print(f"Epoch: {epoch} Epoch loss: {epoch_loss}")

    def encode(self, x):
        return self.session.run(self.h, feed_dict={self.x_in: x})

    def decode(self, x):
        return self.session.run(self.output, feed_dict={self.x_in: x})


def generate_embedded_plane_samples(x1, x2, npoints):
    v = np.vstack([x1, x2])
    c = np.random.normal(size=(npoints, 2))
    z = np.matmul(c, v)  # samples from 2d embedded plane
    # z_lk = C_li v_ik
    return z


def main():

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    x1 = np.array([1, 0, 0])
    x2 = np.array([0, 1, 0])
    z = generate_embedded_plane_samples(x1, x2, npoints=10**4)

    autoencoder = A1(xdim=3, hdim=2)
    autoencoder.train(x_train=z, batch_size=300, n_epochs=400)

    x_sample = np.random.normal(size=(10**1, 3))
    x_out_sample = autoencoder.decode(x_sample)
    projection_error = lambda a, b: ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

    print("-" * 50)
    print("RECONSTRUCTION ~ PCA PROJECTIONS")
    for i in range(x_sample.shape[0]):
        print(f"{x_sample[i]} -> {x_out_sample[i]} prj error: {projection_error(x_sample[i], x_out_sample[i])}")


if __name__ == "__main__":
    main()
