# 한글 임베딩 후 학습시키기
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

# 한글 토큰화 하기
import re # 파이썬에서 정규식 처리용 모듈
from konlpy.tag import Okt # 한글 토큰화
import pandas as pd

# read_csv 할 때는 구분자가 ",", read_table은 구분자가 공백이며 dataframe 타입
train_data  = pd.read_table('./datasets/ratings_data.txt') # ratings보다 용량이 줄어든 버전^^

# 정규 표현식을 써서 한글 외의 문자를 제거한다.
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]","")
#[^A-Za-z] 알파벳 제외하고 -> ^ 제외하고/ A-Z : A부터 Z까진 


# 토큰화
stop_words = ["의","가","은","는"] # 불용어(뺄 단어들), 노가다로 작업 ^^
okt = Okt()
tokenized_data = []
labels = []
for s, label in zip(train_data['document'], train_data['label']): # 한 문장씩 물러오기
    s = str(s) # 첫번째가 nan 값인데 float
    if s!='nan':
        temp_x = okt.morphs(s, stem=True) # 문장을 토큰화한다
        temp_x = [word for word in temp_x if word not in stop_words] # 불용어 목록에 없는것만 저장
        # temp_x : 한 문장
        tokenized_data.append(temp_x)
        labels.append(label)

# 단어 사진 만들기
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_data) # 단어 사전 만들기
print(tokenizer.word_index)

# text를 시퀀스로 바꾸기
sequences = tokenizer.texts_to_sequences(tokenized_data) # 시퀀스로 변경한다

from keras.utils import pad_sequences
max_len=20
data = pad_sequences(sequences, maxlen=max_len, truncating="post")
print(data[:3])

def train_test_split(data, labels, test_size=0.2):
    #데이터를 섞어서 쪼개기 - train_test_split 안쓰고 
    rows_count = data.shape[0] #데이터 개수 저장하기 
    indices = np.arange(rows_count) #0,1,2,3,4,5,,,,, 행개수-1까지 
   
    np.random.shuffle( indices ) # 배열을 받아서 섞어서 준다 
    data = data[indices]
    labels = np.array(labels)
    labels = labels[indices]

    #검증셋
    train_samples = rows_count - int(rows_count*test_size)
    X_train = data[:train_samples]
    y_train = labels[:train_samples]
    X_test = data[train_samples:rows_count]
    y_test = labels[train_samples:rows_count]

    return X_train, np.array(y_train), X_test, np.array(y_test)

X_train,y_train, X_test, y_test = train_test_split(data, labels, test_size=0.2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from gensim.models import KeyedVectors

# ratings로 만들어둔 임베딩 파일 적용
word2vec = KeyedVectors.load_word2vec_format("./myword_vec.bin")
print(word2vec.most_similar("김혜수"))
print(word2vec.most_similar("유해진"))
print(word2vec.most_similar("조승우"))

embedding_dim = 100

word_index = tokenizer.word_index # 커스텀 데이터에 맞춰진 단어 사전
embedding_matrix = np.zeros((10000, embedding_dim)) # 0으로 된 벡터 공간 만들기 
for word in word_index: # 단어 하나씩 가져와서
    if word in word2vec and word_index[word] < 10000:
        embedding_matrix[word_index[word]] = word2vec[word]

model = Sequential()
model.add(Embedding(10000, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#새로 읽어들어온 가중치로 바꿔치기 작업하기 
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable= False 

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train,y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
