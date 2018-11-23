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