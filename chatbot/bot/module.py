stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
from konlpy.tag import Okt
okt = Okt()

class NLP:
    def __init__(self, utterance, model, label_dic, token):
        self.utterance = utterance
        self.model = model
        self.label_dic = label_dic
        self.answer = ""
        self.create_answer(token)

    def create_answer(self, token):
        temp_X = okt.morphs(self.utterance, stem=True)
        temp_X = [word for word in temp_X if not word in stopwords]
        data = ""
        for word in temp_X:
            data += word
            data += " "
        data = data[:-1]
        data_list = [data]
        data_for_model = token.texts_to_matrix(data_list, mode='tfidf').round(2)
        result = self.model.predict(data_for_model, batch_size=32)
        result_list = list(result[0])
        self.answer = self.label_dic[result_list.index(max(result_list))]