#convert each playlist track info to a single sentence
#playlist_name artist_name0 song_name0 ... artist_namei song_namei
#Then train this text via gensim word2vec training
import csv
import pickle
import numpy as np
from gensim.models import Word2Vec
from nltk import ngrams
from sklearn import svm
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
# from keras.layers.embeddings import Embedding

pretty = True
compact = False
cache = {}

def read_text():
    text_data = []
    with open('train_neg_full.txt') as f:
        lines_neg = f.readlines()
        print(len(lines_neg))
        i=0
        for line in lines_neg:
            i+=1
            line = line.replace('#','')
            line = line.replace('\n','')
            line = line.replace('<user>','')
            if(i%10000==0):
                print(i,line)
            text_data.append(line.split(' '))
    i=0
    with open('train_pos_full.txt') as f:
        lines_pos = f.readlines()
        for line in lines_pos:
            i+=1
            line = line.replace('#','')
            line = line.replace('\n', '')
            line = line.replace('<user>', '')
            if(i%10000==0):
                print(i)
            text_data.append(line.split(' '))

    with open('text_data.pkl', 'wb') as file:
        pickle.dump(text_data,file)
    return text_data

def train_word2vec():
    with open('text_data.pkl', 'rb') as file:
        sentences= pickle.load(file)
    print("Training starts")
    model = Word2Vec(sentences, size=200,min_count=5, workers=4, sg=0)
    print("Training ends, saving trained binary file")
    model.save('twitter_word2vec_200.bin')
    #new_model = Word2Vec.load('spotify_model_artist_200.bin')
    #print (new_model)
    return

def process_test():
    vec_model = Word2Vec.load('twitter_word2vec_200.bin')
    X_test = []
    with open('test_data.txt') as f:
        lines_neg = f.readlines()
        print("Total",len(lines_neg))
        i=0
        for line in lines_neg:
            i+=1
            avg_vec=np.zeros(200)
            line = line.replace('#', '')
            line = line.replace('\n', '')
            line = line.replace('<user>', '')

            id = line.split(',', 1)[0]
            line = line.replace(id+',','')

            words = line.split()

            c=0
            counter=0
            for word in words:
                if word in vec_model:
                    avg_vec+=vec_model[word]
                    counter+=1
                c+=1

            if counter>0:
                avg_vec/=counter
                X_test.append(avg_vec)
            else:
                #print("words",words)
                for word in words:
                    three_grams = ngrams(list(word), 3)
                    for gram in three_grams:
                        gram_joined=""
                        gram_joined.join(gram)
                        for letter in gram:
                            gram_joined+=letter
                        if gram_joined in vec_model:
                            avg_vec += vec_model[gram_joined]
                            counter += 1

                    three_grams = ngrams(list(word), 4)
                    for gram in three_grams:
                        gram_joined = ""
                        gram_joined.join(gram)
                        for letter in gram:
                            gram_joined += letter

                        if gram_joined in vec_model:
                            avg_vec += vec_model[gram_joined]
                            counter += 1

                avg_vec/=counter
                X_test.append(avg_vec)

                print("Counter = 0",counter)
    print(len(X_test))
    with open('X_test.pkl', 'wb') as file:
        pickle.dump(X_test, file)

    return

def cnn_nlp():

    return

def rnn_nlp():

    return

def lstm_nlp():

    return

def process_train():
    vec_model = Word2Vec.load('twitter_word2vec_200.bin')

    X_train = []
    y_train= []

    with open('train_neg_full.txt') as f:
        lines_neg = f.readlines()
        print("Total",len(lines_neg))
        i=0
        for line in lines_neg:
            i+=1
            avg_vec=np.zeros(200)
            line = line.replace('#', '')
            line = line.replace('\n', '')
            line = line.replace('<user>', '')
            words = line.split()
            counter=0

            if(i%10000==0):
                print(i)
            for word in words:
                if word in vec_model:
                    avg_vec+=vec_model[word]
                    counter+=1
            if counter>0:
                avg_vec/=counter
                X_train.append(avg_vec)
                y_train.append(0)
            else:
                print("Counter = 0")


    print("negatives are completed")
    i=0
    with open('train_pos_full.txt') as f:
        if (i % 10000 == 0):
            print(i)
        avg_vec = np.zeros(200)
        lines_pos = f.readlines()
        print("Total",len(lines_pos))
        for line in lines_pos:
            i += 1
            line = line.replace('#', '')
            line = line.replace('\n', '')
            line = line.replace('<user>', '')
            words = line.split()
            counter=0
            for word in words:
                if word in vec_model:
                    avg_vec+=vec_model[word]
                    counter+=1
            if counter>0:
                avg_vec/=counter
                X_train.append(avg_vec)
                y_train.append(1)
            else:
                print("Counter = 0",i,line)


    print("positives are completed")
    print(len(X_train),len(y_train))

    with open('X_train.pkl', 'wb') as file:
        pickle.dump(X_train,file)

    with open('y_train.pkl', 'wb') as file:
        pickle.dump(y_train, file)

    return
def k_means():
    with open('X_train.pkl', 'rb') as file:
        X_train = np.asarray(pickle.load(file))
    print("X_train loaded")

    with open('y_train.pkl', 'rb') as file:
        y_train = np.asarray(pickle.load(file))
    print("y_train loaded",y_train.shape)

    with open('X_test.pkl', 'rb') as file:
        X_test = np.asarray(pickle.load(file))
    print("X_test loaded")

    ##X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)

    #print(kmeans.labels_[:20])
    #print(kmeans.labels_[2000000:2000030])

    ##preds = kmeans.predict(X_valid)
    preds_test = kmeans.predict(X_test)
    #print(preds[:20])
    #y_pred = model.predict(X_test)
    #print("predicted",len(y_pred))
    #print(y_pred[:15])
    #predictions = [round(value) for value in y_pred]

    with open('submission_kmeans.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        i=0
        accurates=0
        for prediction in preds_test:
            #if prediction == y_valid[i]:
            #    accurates+=1

            if prediction == 0:
                prediction = -1

            i+=1
            writer.writerow({'Id': i, 'Prediction': prediction})
#    print("Validation accuracy", accurates/len(preds))
#    print(preds_test)
    return

def svm_fit_predict():

    with open('X_train.pkl', 'rb') as file:
        X_train = np.asarray(pickle.load(file))
    print("X_train loaded", (X_train))


    with open('y_train.pkl', 'rb') as file:
        y_train = np.asarray(pickle.load(file))
    print("y_train loaded",y_train.shape)


    with open('X_test.pkl', 'rb') as file:
        X_test = np.asarray(pickle.load(file))

    print("X_test loaded")
    #X_train = normalize(X_train)
    #X_test = normalize(X_test)

    print("normalized")
    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("fit")
    #y_pred = model.predict(X_test)
    #print("predicted",len(y_pred))
    #print(y_pred[:15])
    #predictions = [round(value) for value in y_pred]
    with open('submission_svm.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        i=0
        for prediction in preds:
            i+=1
            if prediction == 0:
                prediction=-1

            writer.writerow({'Id': i, 'Prediction': prediction})

    return

def train_xgb_boosting():
    with open('X_train.pkl', 'rb') as file:
        X_train = np.asarray(pickle.load(file))
    print("X_train loaded", (X_train))

    with open('y_train.pkl', 'rb') as file:
        y_train = np.asarray(pickle.load(file))
    print("y_train loaded",y_train.shape)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

    X_train=xgb.DMatrix(data=X_train,label=y_train)
    X_eval=xgb.DMatrix(data=X_valid,label=y_valid)
    print("y_train loaded")

    with open('X_test.pkl', 'rb') as file:
        X_test = xgb.DMatrix(data=np.asarray(pickle.load(file)),label=np.zeros(10000))
    print("X_test loaded")

    # fit model no training data
    #model = XGBClassifier(num_feature=200,predictor='gpu_predictor',booster= 'gblinear')
    #model.fit(X_train, y_train)

    param = {'silent': 1, 'objective': 'binary:logistic', 'booster': 'gblinear',
             'alpha': 0.0001, 'lambda': 1, 'subsample':0.7}


    watchlist = [(X_valid, 'eval'), (X_train, 'train')]
    num_round = 20
    bst = xgb.train(param, X_train, num_round, watchlist)
    preds = bst.predict(X_test)
    labels = X_test.get_label()




    print("fit")
    #y_pred = model.predict(X_test)
    #print("predicted",len(y_pred))
    #print(y_pred[:15])
    #predictions = [round(value) for value in y_pred]
    with open('submission.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        i=0
        for prediction in preds:
            i+=1
            writer.writerow({'Id': i, 'Prediction': prediction})

    return

#process_train()
#process_test()
#train_xgb_boosting()
#read_text()
svm_fit_predict()
#k_means()
#train_word2vec()