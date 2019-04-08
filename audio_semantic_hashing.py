import numpy as np
import tensorflow as tf
import os
import urllib.request
from collections import Counter

SAMPLE_AUDIO = {
    "harpsichord": ["https://ccrma.stanford.edu/~jos/wav/harpsi-cs.wav",
                    "https://ccrma.stanford.edu/~jos/wav/Harpsichord.wav"],
    "cello": ["https://ccrma.stanford.edu/~jos/wav/cello.wav"],
    "trumpet": ["https://ccrma.stanford.edu/~jos/wav/trumpet.wav"],
    "piano": ["https://ccrma.stanford.edu/~jos/wav/pno-cs.wav"]
}

def filename_to_key(fname):
    for k in SAMPLE_AUDIO.keys():
        links = SAMPLE_AUDIO[k]
        if sum([int(fname in a) for a in links]) > 0:
            return k

def download_audio():
    if not os.path.exists("./wavs"):
        os.mkdir("./wavs")
    for k in SAMPLE_AUDIO.keys():
        links = SAMPLE_AUDIO[k]
        for url in links:
            fname = url.split("/")[-1]
            if not os.path.exists(f"./wavs/{fname}"):
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, filename="./wavs/" + fname)
            else:
                print(f"{fname} already downloaded")
    print("finished downloads")


class SemanticHashing(object):

    def __init__(self, xdim, hdim):

        LAYER_DIMS = [xdim, 20, 10, 10, hdim]

        self.xdim = xdim
        self.hdim = hdim

        self.noise = tf.placeholder('float', shape=[None, hdim])
        self.x_in = tf.placeholder('float', [None, xdim])

        self.encoder_layers = [self.x_in]
        for i in range(len(LAYER_DIMS) - 1):
            if i == len(LAYER_DIMS) - 2:
                layer = tf.nn.sigmoid(tf.add(
                    tf.matmul(self.encoder_layers[i],
                              tf.Variable(tf.random_normal([LAYER_DIMS[i], LAYER_DIMS[i + 1]]))),
                    tf.Variable(tf.random_normal([LAYER_DIMS[i + 1]]))) + self.noise)
                self.encoder_layers.append(layer)
            else:
                layer = tf.add(
                    tf.matmul(self.encoder_layers[i], tf.Variable(tf.random_normal([LAYER_DIMS[i], LAYER_DIMS[i + 1]]))),
                    tf.Variable(tf.random_normal([LAYER_DIMS[i + 1]])))
                self.encoder_layers.append(layer)

        self.h = self.encoder_layers[-1]

        self.decoder_layers = [self.h]
        for i in range(len(LAYER_DIMS) - 1):
            layer = tf.add(
                tf.matmul(self.decoder_layers[i],
                          tf.Variable(tf.random_normal([LAYER_DIMS[::-1][i], LAYER_DIMS[::-1][i + 1]]))),
                tf.Variable(tf.random_normal([LAYER_DIMS[::-1][i + 1]])))
            self.decoder_layers.append(layer)

        self.output = self.decoder_layers[-1]
        self.x_out = tf.placeholder('float', [None, xdim])
        self.loss = tf.reduce_mean(tf.square(self.x_out - self.output))
        self.optimizer = tf.train.AdamOptimizer(0.1).minimize(self.loss)
        self.initializer = tf.global_variables_initializer()
        self.session = tf.Session()

    def train(self, x_train, batch_size, n_epochs):

        self.session.run(self.initializer)
        noise_std = 0.2
        for epoch in range(n_epochs):
            epoch_loss = 0.
            noise_std *= 1.05
            noise = np.random.normal(scale=noise_std, size=(batch_size, self.hdim))
            for i in range(int(x_train.shape[0] / batch_size)):
                batch = x_train[i * batch_size: (i + 1) * batch_size]
                _, batch_loss = self.session.run([self.optimizer, self.loss],
                                                 feed_dict={self.x_in: batch, self.x_out: batch, self.noise: noise})
                epoch_loss += batch_loss
            print(f"Epoch: {epoch}/{n_epochs} Epoch loss: {epoch_loss} Noise std: {noise_std}")
            h_entropy = self.encoded_entropy(x_train)
            print(f"Encoded entropy: {h_entropy}")

    def encode(self, x):
        return self.session.run(self.h, feed_dict={self.x_in: x, self.noise: np.random.normal(0.0, 0.0, size=(x.shape[0], self.hdim))})

    def encoded_entropy(self, x):
        h = self.encode(x)
        bits = [''.join([str(e) for e in b]) for b in h]
        cts = Counter(bits).values()
        ps = [ct * 1.0 / sum(cts) for ct in cts]
        return sum([-p * np.log(p) for p in ps])


    def decode(self, x):
        return self.session.run(self.output, feed_dict={self.x_in: x})


def absoluteFilePaths(directory):
   for dirpath, _, filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def main():

    CHUNK_SIZE = 10000

    import scipy.io.wavfile
    download_audio()

    filenames = list(absoluteFilePaths("./wavs"))
    all_chunks = []
    keys = []
    for fname in filenames:
        rate, numpy_audio = scipy.io.wavfile.read(fname)
        x = numpy_audio.flatten()
        all_chunks += [normalize(x[i: i + CHUNK_SIZE]) for i in range(int(x.shape[0]/CHUNK_SIZE))]
        key = filename_to_key(fname.split("/")[-1])
        keys += [key] * int(x.shape[0]/CHUNK_SIZE)

    x_train = np.vstack(all_chunks)
    print(f"training set shape: {x_train.shape}")
    ash = SemanticHashing(xdim=CHUNK_SIZE, hdim=15)
    ash.train(x_train=x_train, batch_size=300, n_epochs=100)

    N_SAMPLES = 10
    test_indices = np.random.choice(len(all_chunks), N_SAMPLES)
    x_test = np.array(all_chunks)[test_indices]
    encoded_x = ash.encode(x_test)
    bit_seqs = set()
    bitseqmp = {}
    for i, encoded_vec in enumerate(encoded_x):
        key = keys[test_indices[i]]
        bitseq = ''.join([str(int(e)) for e in encoded_vec])
        bit_seqs.add(bitseq)
        try:
            bitseqmp[bitseq]
        except:
            bitseqmp[bitseq] = []
        bitseqmp[bitseq] += [key]
        print(f"{key} -> bit sequence: {bitseq}")
    print(f"{len(bit_seqs)} distinct bit sequences")
    for k in bitseqmp.keys():
        print(f"{k} -> {bitseqmp[k]}")


if __name__ == '__main__':
    main()
