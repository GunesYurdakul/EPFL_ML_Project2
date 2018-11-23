import numpy as np
import pickle
from nltk import ngrams


class DataSet:

    def __init__(self, model):
        self.model = model.get_model()
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
                avg_vec = np.zeros(self.model.get_embedding_size())

                line = line.strip()

                for substr in delete_substr:
                    line = line.replace(substr, '')

                if training == 0:
                    id = line.split(',', 1)[0]
                    line = line.replace(id + ',', '')

                words = line.split()
                counter = 0

                if i % 10000 == 0:
                    print("{} lines processed".format(i))

                for word in words:
                    if word in self.model:
                        avg_vec += self.model[word]
                        counter += 1

                if counter > 0:

                    avg_vec /= counter

                    if training:
                        self.X_train.append(avg_vec)
                        self.y_train.append(label)
                    else:
                        self.X_test.append(avg_vec)

                elif counter == 0 and training == 0:

                    for word in words:
                        three_grams = ngrams(list(word), 3)
                        for gram in three_grams:
                            gram_joined = ""
                            gram_joined.join(gram)
                            for letter in gram:
                                gram_joined += letter
                            if gram_joined in self.model:
                                avg_vec += self.model[gram_joined]
                                counter += 1

                        three_grams = ngrams(list(word), 4)
                        for gram in three_grams:
                            gram_joined = ""
                            gram_joined.join(gram)
                            for letter in gram:
                                gram_joined += letter

                            if gram_joined in self.model:
                                avg_vec += self.model[gram_joined]
                                counter += 1

                    avg_vec /= counter
                    self.X_test.append(avg_vec)

        print("Set completed.")

    def create_train_test(self, positive_training, negative_training, delete_substr, training):

        self.__insert_data(positive_training, delete_substr, label=1)
        self.__insert_data(negative_training, delete_substr, label=0)

        if training:
            self.X_train = np.array(self.X_train)
            self.y_train = np.array(self.y_train)

            print('X_train shape: {}'.format(self.X_train.shape))
            print('y_train shape: {}'.format(self.y_train.shape))

            with open('X_train.pkl', 'wb') as file:
                pickle.dump(self.X_train, file)

            with open('y_train.pkl', 'wb') as file:
                pickle.dump(self.y_train, file)

        else:
            self.X_test = np.array(self.X_test)
            self.y_test = np.array(self.y_test)

            print('X_test shape: {}'.format(self.X_test.shape))
            print('y_test shape: {}'.format(self.y_test.shape))

            with open('X_test.pkl', 'wb') as file:
                pickle.dump(self.X_test, file)

            with open('y_test.pkl', 'wb') as file:
                pickle.dump(self.y_test, file)