import numpy as np
from nltk import ngrams
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding

class DataSet:

    def __init__(self, model):
        self.vec_type = model
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.X_train_cnn = []
        self.y_train_cnn = []
        self.X_test_cnn = []

    def __insert_data(self, tweets, label, training):

        i = 0

        for line in tweets:

            i += 1
            avg_vec = np.zeros(self.vec_type.get_embedding_size())

            counter = 0

            if i % 10000 == 0:
                print("{} lines processed".format(i))

            words=[]
            for word in line:
                if word in self.vec_type.get_model():
                    avg_vec += self.vec_type.get_model()[word]
                    counter += 1
                    words.append(word)

            if counter > 0:

                avg_vec /= counter

                if training:
                    self.X_train.append(avg_vec)
                    self.y_train.append(label)
                    self.X_train_cnn.append(words)
                else:
                    self.X_test.append(avg_vec)
                    self.X_test_cnn.append(words)

            elif counter == 0 and training is False:

                for word in line:

                    three_grams = ngrams(list(word), 3)

                    for gram in three_grams:
                        gram_joined = ""
                        gram_joined.join(gram)

                        for letter in gram:
                            gram_joined += letter

                        if gram_joined in self.vec_type.get_model():
                            avg_vec += self.vec_type.get_model()[gram_joined]
                            counter += 1
                            words.append(word)

                    three_grams = ngrams(list(word), 4)

                    for gram in three_grams:
                        gram_joined = ""
                        gram_joined.join(gram)

                        for letter in gram:
                            gram_joined += letter

                        if gram_joined in self.vec_type.get_model():
                            avg_vec += self.vec_type.get_model()[gram_joined]
                            counter += 1
                            words.append(word)

                avg_vec /= counter
                self.X_test.append(avg_vec)
                self.X_test_cnn.append(words)

        print("Set completed.")

    def create_embedding(self, embedding_dim, seq_length):

        tokenizer = Tokenizer(num_words=len(self.vec_type.get_model().vocab))
        tokenizer.fit_on_texts(self.X_train_cnn)

        sequences = tokenizer.texts_to_sequences(self.X_train_cnn)
        train_data = pad_sequences(sequences, maxlen=seq_length)

        sequences_test = tokenizer.texts_to_sequences(self.X_test_cnn)  # test
        test_data = pad_sequences(sequences_test, maxlen=seq_length)

        word_index = tokenizer.word_index
        nb_words = min(len(self.vec_type.get_model().vocab), len(word_index)) + 1

        embedding_matrix = np.zeros((nb_words, embedding_dim))

        for word, i in word_index.items():
            if word in self.vec_type.get_model().vocab:
                embedding_matrix[i] = self.vec_type.get_model()[word]

        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        embedding_layer = Embedding(embedding_matrix.shape[0],  # or len(word_index) + 1
                                    embedding_matrix.shape[1],  # or EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=seq_length,
                                    trainable=False)

        return train_data, test_data, embedding_layer

    def create_train_test(self, positive_training, negative_training, training):

        if training:
            self.__insert_data(positive_training, 1, training)
            self.__insert_data(negative_training, 0, training)

            self.X_train = np.array(self.X_train)
            self.X_train_cnn = np.array(self.X_train_cnn)
            self.y_train = np.array(self.y_train)

            print('X_train shape: {}'.format(self.X_train.shape))
            print('y_train shape: {}'.format(self.y_train.shape))

            np.save('X_train', self.X_train)
            np.save('X_train_cnn', self.X_train_cnn)
            np.save('y_train', self.y_train)

        else:
            self.__insert_data(positive_training, 1, False)

            self.X_test = np.array(self.X_test)
            self.X_test_cnn = np.array(self.X_test_cnn)

            print('X_test shape: {}'.format(self.X_test.shape))

            np.save('X_test', self.X_test)
