import pickle
from gensim.models import Word2Vec


class Word2VecModel:

    def __init__(self, vector_size, word_min_count):
        self.dataset = []
        self.vector_size = vector_size
        self.word_min_count = word_min_count
        self.model = None

    def get_embedding_size(self):
        return self.vector_size

    def get_model(self):
        return self.model.wv

    def get_name(self):
        return 'w2v'

    def load_dataset(self, data_path):
        if len(self.dataset) == 0:
            with open(data_path, 'rb') as file:
                self.dataset = pickle.load(file)

    def __read_dataset(self, data_path, delete_substr):
        with open(data_path) as f:
            lines = f.readlines()
            print("Number of tweets: {}".format(len(lines)))

            i = 0

            for line in lines:
                i += 1

                line = line.strip()

                for substr in delete_substr:
                    line = line.replace(substr, '')

                if i % 10000 == 0:
                    print("{} lines processed".format(i))

                self.dataset.append(line.split())

    def read_text(self, negative_datapath, positive_datapath, delete_substr):

        print('Adding negative tweets')
        self.__read_dataset(negative_datapath, delete_substr)
        print('Adding positive tweets')
        self.__read_dataset(positive_datapath, delete_substr)

        with open('dataset.pkl', 'wb') as file:
            pickle.dump(self.dataset, file)

    def train(self, data_path=''):

        if len(self.dataset) == 0:
            self.load_dataset(data_path)

        print("Training starts")
        self.model = Word2Vec(self.dataset, size=self.vector_size, min_count=self.word_min_count, workers=2, sg=0)
        print("Training ends")

        self.model.save('model_word2vec_' + str(self.vector_size) + '.bin')
        print("Word2Vec Model saved")

    def load_model(self, data_path):
        self.model = Word2Vec.load(data_path)
