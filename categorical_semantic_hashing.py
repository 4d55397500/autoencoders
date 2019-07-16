import numpy as np
from semantic_hashing import SemanticHashing

TRAINING_WORDS = ["foo", "bar", "car", "jar"]


def words_to_vectors():
    return np.eye(len(TRAINING_WORDS))


def index_to_word(ix):
    return TRAINING_WORDS[ix]


def categorical_semantic_hashing():

    x_train = np.tile(words_to_vectors(), (10**4, 1))
    ae = SemanticHashing(xdim=x_train.shape[1], hdim=3)
    ae.train(x_train=x_train, batch_size=200, n_epochs=100)

    x_test = words_to_vectors()
    encoded_x = ae.encode(x_test)
    print("---------------------------------------------")
    for i, encoded_vec in enumerate(encoded_x):
        word = index_to_word(i)
        print(f"{word} -> bit sequence: {''.join([str(int(e)) for e in encoded_vec])}")


categorical_semantic_hashing()