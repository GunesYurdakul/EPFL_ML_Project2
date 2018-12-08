import numpy as np
import pickle
from nltk import ngrams
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
import re

class DataSet:

    def __init__(self, model):
        self.vec_type = model
        self.X_train = []
        self.X_train_cnn = []
        self.X_test_cnn = []
        self.y_train_cnn = []
        self.embedding = {}
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.pos_count = {}
        self.neg_count = {}
            
    def __get_weight(self, word):
        
        if word not in self.pos_count:
            positive_count=1
        else:
            positive_count = self.pos_count[word]

        if word not in self.neg_count:
            negative_count=1
        else:
            negative_count = self.neg_count[word]

        if positive_count>negative_count:
            weight=positive_count/negative_count
        else:
            weight=negative_count/positive_count
            
        return weight
    
    def __tokenize(self, line):
        ps = PorterStemmer()
        line=re.sub(r'\d+', '', line)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(line)
        words=[]
        for token in tokens:
            if token in self.vec_type.get_model():
                words.append(ps.stem(token))
        return words
    
    def __preprocess(self,file_path,delete_substr,training,stop_words,label):
        stopWords = set(stopwords.words('english'))
        with open(file_path) as f:
            
            lines = f.readlines()
            print("Total lines (): {}".format(len(lines)))

            i = 0

            for line in lines:

                i += 1

                line = line.strip()

                for substr in delete_substr:
                    line = line.replace(substr, '')

                if training is False:
                    id = line.split(',', 1)[0]
                    line = line.replace(id + ',', '')
                max_weight=1
                if stop_words:
                    words_non_filtered = self.__tokenize(line)
                    words=[]
                    for word in words_non_filtered:
                        #weight=self.__get_weight(word)
                        #if weight> max_weight:
                        #    max_weight = weight
                        #if self.__get_weight(word)>1.5:
                        if word not in stopWords:
                            words.append(word)
                    #if len(words)==0:
                    #    weight=self.__get_weight(word)
                    #    if word not in stopWords:
                    #        words.append(word)
                else:
                    words=self.__tokenize(line)
                if training:
                    if len(words)>0:
                        self.X_train_cnn.append(words)
                        self.y_train_cnn.append(label)
                else:
                    self.X_test_cnn.append(words)
        
                if i % 10000 == 0:
                    print("{} lines processed".format(i)," len words",len(words))

        return
    
    def __insert_data(self, file_path, delete_substr, label, training, use_weight, stop_words):
        
        stopWords = set(stopwords.words('english'))
        
        with open(file_path) as f:

            lines = f.readlines()
            print("Total lines (label = " + str(label) + "): {}".format(len(lines)),use_weight)

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
                
                if stop_words:
                    words_non_filtered = line.split()
                    words=[]
                
                    for word in words_non_filtered:
                        if word not in stopWords:
                            words.append(word)
                else:
                    words=line.split()

                counter = 0


                for word in words:
                    if use_weight>0:
                        weight=self.__get_weight(word)
                    else:
                        weight=1
                        
                    if self.vec_type.get_name() == 'glove':
                        if word in self.vec_type.get_vocab():
                            word = self.vec_type.get_vocab()[word]
                            avg_vec += weight*self.vec_type.get_model()[word]
                            counter += weight

                    else:
                        if word in self.vec_type.get_model():
                            avg_vec += weight*self.vec_type.get_model()[word]
                            counter += weight

                if counter > 0:

                    avg_vec /= counter

                    if training:
                        self.X_train.append(avg_vec)
                        self.y_train.append(label)
                    else:
                        self.X_test.append(avg_vec)

                elif counter == 0 and training is False:

                    for word in words:
                        
                        if use_weight>0:
                            weight=self.__get_weight(word)
                        else:
                            weight=1
                        
                        three_grams = ngrams(list(word), 3)

                        for gram in three_grams:
                            gram_joined = ""
                            gram_joined.join(gram)
                            for letter in gram:
                                gram_joined += letter

                            if self.vec_type.get_name() == 'glove':
                                if gram_joined in self.vec_type.get_vocab():
                                    gram_joined = self.vec_type.get_vocab()[gram_joined]
                                    avg_vec += weight*self.vec_type.get_model()[gram_joined]
                                    counter += weight
                            else:
                                if gram_joined in self.vec_type.get_model():
                                    avg_vec += weight*self.vec_type.get_model()[gram_joined]
                                    counter += weight

                        three_grams = ngrams(list(word), 4)

                        for gram in three_grams:
                            gram_joined = ""
                            gram_joined.join(gram)
                            for letter in gram:
                                gram_joined += letter

                            if self.vec_type.get_name() == 'glove':
                                if gram_joined in self.vec_type.get_vocab():
                                    gram_joined = self.vec_type.get_vocab()[gram_joined]
                                    avg_vec += weight*self.vec_type.get_model()[gram_joined]
                                    counter += weight
                            else:
                                if gram_joined in self.vec_type.get_model():
                                    avg_vec += weight*self.vec_type.get_model()[gram_joined]
                                    counter += weight

                    avg_vec /= counter
                    self.X_test.append(avg_vec)
                    
                if i % 10000 == 0:
                    print("{} lines processed".format(i)," len words",len(words)," ",counter)
                    

        print("Set completed.")
        
    def create_count(self, file_path,delete_substr,is_positive,weight):
        
        with open(file_path) as f:
            lines = f.readlines()
            print("Total lines (label = " + str(is_positive) + "): {}".format(len(lines)))
            i = 0

            for line in lines:

                i += 1
                avg_vec = np.zeros(self.vec_type.get_embedding_size())

                line = line.strip()

                for substr in delete_substr:
                    line = line.replace(substr, '')

                words = line.split()
                
                for word in words:
                    if (is_positive):
                        if word in self.pos_count:
                            self.pos_count[word]+=1
                        else:
                            self.pos_count[word]=1
                    else:
                        if word in self.neg_count:
                            self.neg_count[word]+=1
                        else:
                            self.neg_count[word]=1
            
                if i % 10000 == 0:
                    print("{} lines processed".format(i))
        if weight>0:
            if (is_positive):
                data=np.asarray(list(self.pos_count.values())).reshape(-1, 1)
                scaler = MinMaxScaler(copy=True, feature_range=(1, weight))
                scaler.fit(data)
                normalized = scaler.transform(data)
                i=0
                for key in self.pos_count:
                    self.pos_count[key]=normalized[i][0]
                    i+=1
            else:
                data=np.asarray(list(self.neg_count.values())).reshape(-1, 1)
                scaler = MinMaxScaler(copy=True, feature_range=(1, weight))
                scaler.fit(data)
                normalized = scaler.transform(data)
                i=0
                for key in self.pos_count:
                    self.neg_count[key]=normalized[i][0]
                    i+=1

        return
    def create_embedding(self, embedding_dim, seq_length):
        
        tokenizer = Tokenizer(num_words=len(self.vec_type.model.wv.vocab))
        tokenizer.fit_on_texts(self.X_train_cnn)
        
        sequences = tokenizer.texts_to_sequences(self.X_train_cnn)
        train_data = pad_sequences(sequences, maxlen=seq_length)
        
        sequences_test = tokenizer.texts_to_sequences(self.X_test_cnn)#test
        test_data = pad_sequences(sequences_test, maxlen=seq_length)

        word_index = tokenizer.word_index
        nb_words = max(len(self.vec_type.model.wv.vocab), len(word_index))+1

        embedding_matrix = np.zeros((nb_words, embedding_dim))
        
        for word, i in word_index.items():
            if word in self.vec_type.model.wv.vocab:
                embedding_matrix[i] = self.vec_type.model.wv[word]
                
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        embedding_layer = Embedding(embedding_matrix.shape[0], # or len(word_index) + 1
                                    embedding_matrix.shape[1], # or EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=seq_length,
                                    trainable=False)
        return train_data, test_data, embedding_layer
        
    def create_train_test(self, positive_training, negative_training, delete_substr, training,cnn,weight,stop_words,model='w2v'):


        if model == 'glove':
            vocab = self.vec_type.get_vocab()
        if cnn:
            
            if training:
                self.__preprocess(positive_training, delete_substr, training, stop_words,1)
                self.__preprocess(negative_training, delete_substr, training, stop_words,0)
                
                self.X_train_cnn = np.array(self.X_train_cnn)
                self.y_train_cnn = np.array(self.y_train_cnn)

                print('X_train shape: {}'.format(self.X_train_cnn.shape))
                print('y_train shape: {}'.format(self.y_train_cnn.shape))

                np.save('X_train_cnn', self.X_train_cnn)
                np.save('y_train_cnn', self.y_train_cnn)

            else:
                self.__preprocess(positive_training, delete_substr, training, stop_words,1)

                self.X_test_cnn = np.array(self.X_test_cnn)
                print('X_test shape: {}'.format(self.X_test_cnn.shape))
                np.save('X_test_cnn', self.X_test_cnn)        
        else:
            if training:
                self.X_train = []
                self.y_train = []
                self.pos_count = {}
                self.neg_count = {}
                self.create_count(positive_training, delete_substr, True,weight)
                self.create_count(negative_training, delete_substr, False,weight)

                self.__insert_data(positive_training, delete_substr, 1, training, weight, stop_words)
                self.__insert_data(negative_training, delete_substr, 0, training, weight, stop_words)

                self.X_train = np.array(self.X_train)
                self.y_train = np.array(self.y_train)

                print('X_train shape: {}'.format(self.X_train.shape))
                print('y_train shape: {}'.format(self.y_train.shape))

                np.save('X_train', self.X_train)
                np.save('y_train', self.y_train)
                np.save('pos_count', self.pos_count)
                np.save('neg_count', self.neg_count)

            else:
                self.X_test = []
                self.__insert_data(positive_training, delete_substr, 1, training,   use_weight= weight, stop_words= stop_words)

                self.X_test = np.array(self.X_test)

                print('X_test shape: {}'.format(self.X_test.shape))

                np.save('X_test', self.X_test)