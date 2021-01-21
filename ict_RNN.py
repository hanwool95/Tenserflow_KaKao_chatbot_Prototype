from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Embedding, Dense, SimpleRNN

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import csv, codecs, pickle

import pandas as pd

import numpy as np

with codecs.open('proto3.csv', 'r') as f:
    rdr = csv.reader(f)
    num_classes = 6

    x_texts = []
    y_texts = []

    for line in rdr:
        x_texts.append(line[1])
        y_texts.append(line[2])


#texts = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']

data = pd.DataFrame(x_texts, columns = ['text'])
data['label'] = pd.Series(y_texts)


token = Tokenizer()
token.fit_on_texts(data['text'])
print(token.word_index)

X_train = token.texts_to_matrix(data['text'], mode = 'tfidf').round(2)

index_dic = token.index_word

with open('token.pickle','wb') as fw:
    pickle.dump(token, fw)
    print("dumping complete")


y_train = to_categorical(data['label'], num_classes)
print(y_train)

print('훈련 샘플 본문의 크기 : {}'.format(X_train.shape))
print('훈련 샘플 레이블의 크기 : {}'.format(y_train.shape))

"""
model = Sequential()
model.add(Embedding(10000, 4))
model.add(GRU(2, input_shape=(4, 2)))
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
"""

#총 238개의 학습 단어 확인.
max_features = 1000

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='softmax'))
#model.add(Dense(6, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
model.fit(X_train, y_train, epochs=3, batch_size=32)

model.save("ict_neural_model2.h5")



#from tensorflow.keras.models import load_model
#model = load_model("ict_neural_model.h5")