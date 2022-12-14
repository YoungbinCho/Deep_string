# 미리 자주 쓰이는 단어들을 임베딩한 것으로 학습 후
# 사용자의 데이터 분류하기
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

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
    #데이터를 섞어서 쪼개기 - train_test_split 안쓰고 
    rows_count = 200  #data.shape[0] #데이터 개수 저장하기 
    indices = np.arange(data.shape[0]) #0,1,2,3,4,5,,,,, 행개수-1까지 
    # print("섞기전")
    # print( indices[:20] )

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

# 임베딩 파일 받아오기
glove_dir = './datasets/glove.6B'

embedding_index = {} # dict 타입 변수 만들어주기
f = open(glove_dir+"/glove.6B.100d.txt", encoding='utf-8')
lines = f.readlines() # 텍스트 파일을 모두 읽음
for line in lines: # 파일 한줄 마다~
    values = line.split() # 데이터 값들이 공백으로 구분되어 있으므로 공백을 기준으로 토큰을 자른다
    word = values[0] # dog 8.322 9.911 3.23334 
    coefs = np.array(values[1:], dtype="float32") # 단어 이름 다음에 공간 정보가 적혀있음.
    embedding_index[word] = coefs # 단어 : 공간정보 형태로 저장

# print(embedding_index)

# dict 형태에 맞추어 matrix로 전환
# 파일 이름이 glove.6B.100d 이면 100 디멘션
# 이미 임베딩 된 단어의 벡터 정보를 가져와 커스텀 데이터의 시퀀스를 대체
embedding_dim = 100
embedding_matrix = np.zeros((10000, embedding_dim)) # 10000 by 100의 임베딩 공간
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    # tokenizer에 등록된 단어를 꺼내서 embedding_index의 key와 일치하면 embedding vector에 값을 저장
    if i<10000 : # embedding_index에 단어가 없을 수도 있으므로
        if embedding_vector is not None: # 단어가 있다면
            embedding_matrix[i] = embedding_vector

print(embedding_matrix[:5])

model = Sequential()
model.add(Embedding(10000, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#새로 읽어들어온 가중치로 바꿔치기 작업하기 
model.layers[0].set_weights([embedding_matrix]) # 임베딩층을 아까 만든 임베딩 메트릭스로 대체
model.layers[0].trainable= False # Embedding 층은 막아 놓고 Dense층만...
# 즉, 임베딩 층에 없는 단어는 학습이 안됨!

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train,y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 콜백으로 저장해도 되고
model.save_weights('imdb_model.h5') # h5 - 대용량 파일 저장 확장자

import pickle
f = open("embeding.hist",mode='wb') # 객체를 저장할 때는 binary로 저장해야한다
pickle.dump(history.history, file=f)
f.close()

f = open("embeding.hist", mode='rb')
history = pickle.load(f)
f.close()

import matplotlib.pyplot as plt
def drawChart(history):
    acc = history['acc'] # 컴파일 시 metrics=['acc'] 약어로 받아왔으므로 약어로 뽑아오기
                         # metrics=['accuracy'] -> accuracy
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1,len(acc)+1) # x축
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title("Training and Validation Accuracy")
    plt.show()

    plt.figure() # 리프레쉬
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title("Training and Validation loss")
    plt.show()
    
drawChart(history)