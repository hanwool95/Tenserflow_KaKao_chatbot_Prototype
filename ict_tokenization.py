import pandas as pd
from konlpy.tag import Okt


train_data = pd.read_table('ict_utf.txt', error_bad_lines=False)

print(train_data[:5])
print(train_data.isnull().values.any())
train_data = train_data.dropna(how = 'any')
print(train_data[:5])

train_data.columns = ['id', 'doc']


train_data['doc'] = train_data['doc'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

okt = Okt()
tokenized_data = []
for sentence in train_data['doc']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    tokenized_data.append(temp_X)

"""
채팅 관련 시각화.
print('채팅의 최대 길이 :',max(len(l) for l in tokenized_data))
print('채팅의 평 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
"""

from gensim.models import Word2Vec
model = Word2Vec(sentences = tokenized_data, size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

print(model.wv.most_similar("창업"))

print(model.wv.most_similar("개발"))

print(model.wv.most_similar("캠프"))

print(model.wv.most_similar("사무실"))

model.wv.save_word2vec_format('ict_model') # 모델 저장

#loaded_model = KeyedVectors.load_word2vec_format("ict_model") # 모델 로드