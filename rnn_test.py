from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Embedding, Dense, SimpleRNN

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from konlpy.tag import Okt

import csv, codecs, pickle

import pandas as pd

import numpy as np

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

answer_dic = {0: '월요일 9시입니다.', 1: '054-260-1877입니다.', 2: '고민 언제든지 들어드릴게요.', 3: '심심하면 놀아드릴게요.',
              4: 'OO시에 헤브론홀 문을 닫습니다.', 5: '헤브론홀 사용후 정리 잘해주세요.'}

from tensorflow.keras.models import load_model
model = load_model("ict_neural_model2.h5")

with open('token.pickle', 'rb') as fr:
    train_data = pickle.load(fr)
    print(train_data.word_index)

Flag = True
while Flag:
    text = input("입력해주세요")

    okt = Okt()
    temp_X = okt.morphs(text, stem=True)
    temp_X = [word for word in temp_X if not word in stopwords]
    data = ""
    for word in temp_X:
        data += word
        data += " "
    data = data[:-1]
    #print(data)
    data_list = [data]
    data_for_model = train_data.texts_to_matrix(data_list, mode = 'tfidf').round(2)
    #print(data_for_model)
    result = model.predict(data_for_model, batch_size=32)
    #print(result)
    result_list = list(result[0])
    print(result_list)
    print(answer_dic[result_list.index(max(result_list))])




