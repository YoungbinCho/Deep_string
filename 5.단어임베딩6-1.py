# # word2vec으로 한글 임베딩 만들기
# # https://radimrehurek.com/gensim/models/word2vec.html
# # pip install konlpy
# # pip install gensim

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

"""
model = Word2Vec(sentences="토큰화된단어", vector_size=100, window=5,
    min_count=1, workers=4)
model.wv.save_word2vec_format("파일명")
"""

# 한글 토큰화 하기
import re # 파이썬에서 정규식 처리용 모듈
from konlpy.tag import Okt
import pandas as pd

train_data  = pd.read_table('./datasets/ratings.txt')

# 정규 표현식을 써서 한글 외의 문자를 제거한다.
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]","")
#[^A-Za-z] 알파벳 제외하고 -> ^ 제외하고/ A-Z : A부터 Z까진 


# 토큰화
stop_words = ["의","가","은","는"] # 불용어(뺄 단어들)
okt = Okt()
tokenized_data = []
for s in train_data['document']: # 한 문장씩 물러오기
    s = str(s) # 첫번째가 nan 값인데 float
    if s!='nan':
        temp_x = okt.morphs(s, stem=True) # 문장을 토큰화한다
        temp_x = [word for word in temp_x if word not in stop_words] # 불용어에 없는것만 저장
        tokenized_data.append(temp_x)

print( tokenized_data[:5] )

model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
model.wv.save_word2vec_format("myword_vec.bin")

from gensim.models import KeyedVectors
models = KeyedVectors.load_word2vec_format('myword_vec.bin')
print(models.most_similar("최민식"))
