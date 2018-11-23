# DEPRECATED

pretty = True
compact = False
cache = {}

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



"""
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
"""