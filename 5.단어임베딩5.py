# 구글에서 만든 word2vec 사용!
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras import models, layers
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
import pickle

def loadData():
    f = open("imdb_data.bin","rb")
    data2 = pickle.load(file=f)
    f.close()
    train_data = data2["train_data"]
    train_labels = data2["train_labels"]
    test_data = data2["test_data"]
    test_labels = data2["test_labels"]
    return train_data, train_labels, test_data, test_labels

train_data, train_labels, test_data, test_labels = loadData()
print(len(train_data), len(train_labels), len(test_data),len(test_labels))
print(train_data[:2])
print(train_labels[:2])
print(test_data[:2])
print(test_labels[:2])

# 토큰나이저-단어별 인덱스 만들어주는- 객체 생성(단어 종류는 10000으로 한정)
tokenizer = Tokenizer(num_words=10000)

# 단어 사전 만들기
tokenizer.fit_on_texts(train_data)
print(tokenizer.word_index) 
print(tokenizer.word_counts) 

# 문자열을 시퀀스로 변경하기
sequences = tokenizer.texts_to_sequences(train_data)

from keras.utils import pad_sequences
max_len = 20
data = pad_sequences(sequences, maxlen=max_len, truncating='post')

import numpy as np
def train_test_split(data, labels, test_size=0.2):
    rows_count = 200  #data.shape[0] #데이터 개수 저장하기 
    indices = np.arange(data.shape[0]) #0,1,2,3,4,5,,,,, 행개수-1까지 


    #인덱스를 섞는다 
    np.random.shuffle( indices ) # 배열을 받아서 섞어서 준다 
    print("섞기후")
    print( indices[:20] )
    data = data[indices]
    labels = np.array(labels)
    labels = labels[indices]
    #print("데이터 전체 개수 ", rows_count)

    #검증셋
    train_sampls = rows_count - int(rows_count*test_size)
    X_train = data[:train_sampls]
    y_train = labels[:train_sampls]
    X_test = data[train_sampls:rows_count]
    y_test = labels[train_sampls:rows_count]

    return X_train, np.array(y_train), X_test, np.array(y_test)

X_train,y_train, X_test, y_test = train_test_split(data, train_labels, test_size=0.2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from gensim.models import KeyedVectors
# GoogleNews-vectors-negative300 -> 디멘션 300
word2vec = KeyedVectors.load_word2vec_format("./datasets/GoogleNews-vectors-negative300.bin", binary=True)
print(word2vec.most_similar("king"))
print(word2vec.most_similar("pretty"))
print(word2vec.most_similar("accident"))

# word_index    "the":1, "like":2
print(word2vec["school"]) # 연산자 중복 기능 - 연산자를 마치 함수처럼 잡아채서 새로운 기능 부여 -> c++, c#, 파이썬에서만
# {"school":[1.2 0.8 0.69 .......]}

embedding_dim = 300

word_index = tokenizer.word_index # 커스텀 데이터에 맞춰진 단어 사전
embedding_matrix = np.zeros((10000, embedding_dim)) # 0으로 된 벡터 공간 만들기 
for word in word_index: # 단어 하나씩 가져와서
    # print(word)
    # 그 단어가 word2vec에 있을 때 + word_index[word]의 값, 즉 워드 인덱스 값이 10000보다 작을 때(근데이미 토큰나이저를 10000으로 한정했기 때문에 안넘을거같은데..)  
    # ex) word_index["school"] = 1 -> embedding matrix 1행에  word2vec가 가진 "school"의 공간정보가 들어감
    if word in word2vec and word_index[word] < 10000:
        embedding_matrix[word_index[word]] = word2vec[word]

print( embedding_matrix[:5,:30])

# model = Sequential()
# model.add(Embedding(10000, embedding_dim, input_length=max_len))
# model.add(Flatten())
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# #새로 읽어들어온 가중치로 바꿔치기 작업하기 
# model.layers[0].set_weights([embedding_matrix])
# model.layers[0].trainable= False # Embedding 층은 막아 놓고 Dense층만...
# # 즉, 임베딩 층에 없는 단어는 학습이 안됨!

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(X_train,y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# # 콜백으로 저장해도 되고
# model.save_weights('imdb_model.h5') # h5 - 대용량 파일 저장 확장자

# import pickle
# f = open("embeding.hist",mode='wb') # 객체를 저장할 때는 binary로 저장해야한다
# pickle.dump(history.history, file=f)
# f.close()

# f = open("embeding.hist", mode='rb')
# history = pickle.load(f)
# f.close()

# import matplotlib.pyplot as plt
# def drawChart(history):
#     acc = history['acc'] # 컴파일 시 metrics=['acc'] 약어로 받아왔으므로 약어로 뽑아오기
#                          # metrics=['accuracy'] -> accuracy
#     val_acc = history['val_acc']
#     loss = history['loss']
#     val_loss = history['val_loss']

#     epochs = range(1,len(acc)+1) # x축
    
#     plt.plot(epochs, acc, 'bo', label='Training acc')
#     plt.plot(epochs, val_acc, 'b', label='Validation acc')
#     plt.title("Training and Validation Accuracy")
#     plt.show()

#     plt.figure() # 리프레쉬
#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     plt.title("Training and Validation loss")
#     plt.show()
    
# drawChart(history)