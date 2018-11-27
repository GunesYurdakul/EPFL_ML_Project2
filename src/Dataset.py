import numpy as np
import pickle
from nltk import ngrams


class DataSet:

    def __init__(self, model):
        self.vec_type = model
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def __insert_data(self, file_path, delete_substr, label, training):

        with open(file_path) as f:

            lines = f.readlines()
            print("Total lines (label = " + str(label) + "): {}".format(len(lines)))

            i = 0

            for line in lines:

                i += 1
                avg_vec = np.zeros(self.vec_type.get_embedding_size())

                line = line.strip()

                for substr in delete_substr:
                    line = line.replace(substr, '')

                if training is False:
                    id = line.split(',', 1)[0]
                    line = line.replace(id + ',', '')

                words = line.split()
                counter = 0

                if i % 10000 == 0:
                    print("{} lines processed".format(i))

                for word in words:

                    if self.vec_type.get_name() == 'glove':
                        if word in self.vec_type.get_vocab():

                            word = self.vec_type.get_vocab()[word]
                            avg_vec += self.vec_type.get_model()[word]
                            counter += 1

                    else:
                        if word in self.vec_type.get_model():
                            avg_vec += self.vec_type.get_model()[word]
                            counter += 1

                if counter > 0:

                    avg_vec /= counter

                    if training:
                        self.X_train.append(avg_vec)
                        self.y_train.append(label)
                    else:
                        self.X_test.append(avg_vec)

                elif counter == 0 and training is False:

                    for word in words:

                        three_grams = ngrams(list(word), 3)

                        for gram in three_grams:
                            gram_joined = ""
                            gram_joined.join(gram)
                            for letter in gram:
                                gram_joined += letter

                            if self.vec_type.get_name() == 'glove':
                                if gram_joined in self.vec_type.get_vocab():
                                    gram_joined = self.vec_type.get_vocab()[gram_joined]
                                    avg_vec += self.vec_type.get_model()[gram_joined]
                                    counter += 1
                            else:
                                if gram_joined in self.vec_type.get_model():
                                    avg_vec += self.vec_type.get_model()[gram_joined]
                                    counter += 1

                        three_grams = ngrams(list(word), 4)

                        for gram in three_grams:
                            gram_joined = ""
                            gram_joined.join(gram)
                            for letter in gram:
                                gram_joined += letter

                            if self.vec_type.get_name() == 'glove':
                                if gram_joined in self.vec_type.get_vocab():
                                    gram_joined = self.vec_type.get_vocab()[gram_joined]
                                    avg_vec += self.vec_type.get_model()[gram_joined]
                                    counter += 1
                            else:
                                if gram_joined in self.vec_type.get_model():
                                    avg_vec += self.vec_type.get_model()[gram_joined]
                                    counter += 1

                    avg_vec /= counter
                    self.X_test.append(avg_vec)

        print("Set completed.")

    def create_train_test(self, positive_training, negative_training, delete_substr, training, model='w2v'):

        if model == 'glove':
            vocab = self.vec_type.get_vocab()

        if training:
            self.__insert_data(positive_training, delete_substr, 1, training)
            self.__insert_data(negative_training, delete_substr, 0, training)

            self.X_train = np.array(self.X_train)
            self.y_train = np.array(self.y_train)

            print('X_train shape: {}'.format(self.X_train.shape))
            print('y_train shape: {}'.format(self.y_train.shape))

            np.save('X_train', self.X_train)
            np.save('y_train', self.y_train)

        else:
            self.__insert_data(positive_training, delete_substr, 1, training)

            self.X_test = np.array(self.X_test)

            print('X_test shape: {}'.format(self.X_test.shape))

            np.save('X_test', self.X_test)