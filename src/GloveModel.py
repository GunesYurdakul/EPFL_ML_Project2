import pickle
import numpy as np


class GloveModel:

    def __init__(self, cooc_path, embedding_dim=20, eta=0.001, alpha= 0.75, epochs=10, nmax=100):
        print("loading cooccurrence matrix")

        with open(cooc_path, 'rb') as f:
            self.cooc_matrix = pickle.load(f)

        self.embedding_dim = embedding_dim
        self.eta = eta
        self.alpha = alpha
        self.epochs = epochs
        self.nmax = nmax
        self.model = None

        with open('../helpers/vocab.pkl', 'rb') as file:
            self.vocab = pickle.load(file)

    def get_name(self):
        return 'glove'

    def get_model(self):
        return self.model

    def train(self):

        print("{} nonzero entries".format(self.cooc_matrix.nnz))
        print("using nmax =", self.nmax, ", cooc.max() =", self.cooc_matrix.max())
        print("initializing embeddings")

        xs = np.random.normal(size=(self.cooc_matrix.shape[0], self.embedding_dim))
        ys = np.random.normal(size=(self.cooc_matrix.shape[1], self.embedding_dim))

        for epoch in range(self.epochs):
            print("epoch {}".format(epoch))
            for ix, jy, n in zip(self.cooc_matrix.row, self.cooc_matrix.col, self.cooc_matrix.data):
                logn = np.log(n)
                fn = min(1.0, (n / self.nmax) ** self.alpha)
                x, y = xs[ix, :], ys[jy, :]
                scale = 2 * self.eta * fn * (logn - np.dot(x, y))
                xs[ix, :] += scale * y
                ys[jy, :] += scale * x

        np.save('word_embeddings_glove', xs)

    def load_model(self, data_path):
        self.model = np.load(data_path)

    def get_vocab(self):
        return self.vocab


