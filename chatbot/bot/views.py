from django.http import HttpResponse, JsonResponse
import json, codecs, csv, pickle

from tensorflow.keras.models import load_model
from konlpy.tag import Okt

from .module import *


def create_label_dict():
    with codecs.open('label.csv', 'r') as f:
        rdr = csv.reader(f)
        next(rdr)

        label_dict = {}

        for line in rdr:
            if line[2] in label_dict.keys():
                pass
            else:
                label_dict[int(line[2])] = line[3]
    return label_dict


def load_token():
    with open('token.pickle', 'rb') as fr:
        token = pickle.load(fr)
        #print(token.word_index)
    return token


def kakao_simpleText(text):
    return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "basicCard": {
                            "description": text
                        }
                    }
                ]
            }
        }

print('loading model')
model = load_model("ict_neural_model2.h5")
print('loading dictionary')
label_dict = create_label_dict()
print(label_dict)
print('loading token')
token = load_token()


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def hello(request):
    json_body = json.loads(request.body)
    user_utterance = json_body['userRequest']['utterance']

    nlp_object = NLP(user_utterance, model, label_dict, Okt, token)

    text = kakao_simpleText(nlp_object.answer)
    return JsonResponse(text)