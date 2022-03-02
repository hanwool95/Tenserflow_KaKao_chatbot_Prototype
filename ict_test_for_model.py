from konlpy.tag import Okt
from gensim.models import KeyedVectors


loaded_model = KeyedVectors.load_word2vec_format("ict_model") # 모델 로드

import csv, copy, codecs

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

with codecs.open('proto1.csv', 'r') as f:
    rdr = csv.reader(f)
    next(rdr)
    read_list = []
    for line in rdr:
        read_list.append(line)

f = open('proto2.csv', 'w', newline="")
wr = csv.writer(f)
count = 1
for line in read_list:
    wr.writerow([count, line[1], line[2]])
    count += 1
    okt = Okt()
    temp_X = okt.morphs(line[1], stem=True)
    temp_X = [word for word in temp_X if not word in stopwords]

    word_list = []
    for i in range(len(temp_X)):
        if word_list == []:
            word_list.append(temp_X[i])
            try:
                similar_word_list = loaded_model.wv.most_similar(temp_X[i])
                for morph in similar_word_list:
                        word_list.append(morph[0])
            except:
                pass
        else:
            current_word_list = copy.deepcopy(word_list)
            for j in range(len(word_list)):
                word_list[j] = word_list[j]+ " " + temp_X[i]
            try:
                similar_word_list = loaded_model.wv.most_similar(temp_X[i])
                for morph in similar_word_list:
                    for j in range(len(current_word_list)):
                        word_list.append(current_word_list[j] + " " + morph[0])
            except:
                pass
    for word in word_list:
        print(word)
        wr.writerow([count, word, line[2]])
        count += 1
f.close()
