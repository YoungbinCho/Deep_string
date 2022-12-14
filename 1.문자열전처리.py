import numpy as np
samples = [
    "The cat sat on the mat",
    "The dog ate my homework",
    "The cat is very cute",
    "THe fox is very smart and very beautiful and follow"
]

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer # 토큰으로 만들어주는 라이브러리

# 가장 빈도가 높은 1000개의 단어만 선택하도록한다
tokenizer = Tokenizer(num_words=5) # 5개만 ^^ num_words: 단어의 종류 개수 
# 단어 인덱스를 구축한다. {"the":1, "cat:2", ....}
tokenizer.fit_on_texts(samples) #  텍스트를 토큰화 해서 인덱스를 부여한다, 리스트 형태로 넣어주기! 

# 원핫인코딩을 하지 않고 임베딩 할 때 거치는 과정!
# 시퀀스를 가져온다 -> 각 문장을 인덱스로 바꾸어서 들고옴
sequences = tokenizer.texts_to_sequences(samples)
print(sequences)
# [[1, 3, 6, 7, 1, 8], [1, 9, 10, 11, 12], [1, 3, 4, 2, 13], [1, 14, 4, 2, 15, 5, 2, 16, 5, 17]]
# 분류형 데이터들 1,2,3,4,5,6,7,8,9,10  이렇게 해 놓으면 숫자가 높은 것들이 영향을 더 많이 미침
# 원핫인코딩을 사용해서 다 대등한 가치임을 나타낸다.

# 원핫인코딩(texts_to_matrix)
# 인덱스를 열로 만들어 1,2,3,4..번째 단어가 있는지 없는지 0 1 0 1 이렇게 나타냄
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary') 
# binary는 단어 존재 유무만 표시 count는 횟수
print(one_hot_results.shape)
print(one_hot_results)

# 희소행렬(sparse matrix) : 행렬의 요소 대부분이 0이고 어쩌다 1이 하나씩 등장
# ex) num_words=10000일 경우 포함되지 않는 단어 대부분이 0..
# -> 원핫 해싱(해시를 사용한다) => 밀집벡터(단어와 단어의 거리를 가지고 데이터저장)
# 케라스가 Embedding이라는 계층을 주고, Embedding 층에서 단어 사이의 거리를 학습시킨다.
# 워드투벡터(구글), Bert, ....
# 최신 : TextVectorization, text_dataset_from_directory