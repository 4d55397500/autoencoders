import numpy as np
from semantic_hashing import SemanticHashing

TRAINING_PHRASES = [
    "this is a cat",
    "this is my hat",
    "this is a hat",
    "this is a big hat",
    "this is one fat cat",
    "this is a smelly rat",
    "that is completely random",
    "like night versus day"
]

def phrases_to_vectors():
    # create bag of words representation
    words = sorted(list(set([w for phrase in TRAINING_PHRASES for w in phrase.split(" ")])))
    vecs = []
    for phrase in TRAINING_PHRASES:
        pv = [0] * len(words)
        for w in phrase.split(" "):
            pv[words.index(w)] += 1
        vecs.append(pv)
    return np.array(vecs).reshape((len(TRAINING_PHRASES), len(words)))

def index_to_phrase(ix):
    return TRAINING_PHRASES[ix]

def bag_of_words_semantic_hashing():
    x_train = np.tile(phrases_to_vectors(), (10**4, 1))
    ae = SemanticHashing(xdim=x_train.shape[1], hdim=5)
    ae.train(x_train=x_train, batch_size=200, n_epochs=100)

    x_test = phrases_to_vectors()
    encoded_x = ae.encode(x_test)
    print("---------------------------------------------")
    for i, encoded_vec in enumerate(encoded_x):
        phrase = index_to_phrase(i)
        print(f"{phrase} -> bit sequence: {''.join([str(int(e)) for e in encoded_vec])}")


bag_of_words_semantic_hashing()


