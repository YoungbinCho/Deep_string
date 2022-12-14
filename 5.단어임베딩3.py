# 텍스트 데이터를 받아 시퀀스로 직접 만들어보기..!

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras import models, layers
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

#1. 데이터 가져오기
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
print(X_train.shape)
print(y_train.shape)
# print(X_train[0]) # [1, 14, 22, 16, 43, 530, ...  이렇게 숫자로 되어있음

# 원본 파일 받아오기!!
#1. 텍스트 파일을 읽고 라벨링화 하기
#2. 받아온 시퀀스를 역으로 문자열로 바꿔서 토큰나이즈
# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz


#1. 파일을 읽고, 라벨링한다. 문자열은 문자열대로 덧붙이고, 라벨은 라벨대로
#2. 인덱싱화(시퀀스를 만들고)
#3. 원핫인코딩하고 학습하기
import os, shutil, pathlib

def makeData(foldername='train'):
    # 해당 데이터셋은 train, test가 하위폴더로 나뉘어잇으므로
    # 폴더이름만 바꿔주면 train, test 데이터셋을 따로저장할 수 있다
    folder_url = pathlib.Path('./datasets/aclImdb')
    texts=[]
    labels=[]
    #train -> neg,pos
    #test -> neg,pos
    #neg 먼저 읽고
    negDir = folder_url/foldername/"neg"
    for filename in os.listdir(negDir):
        f = open(negDir/filename, 'r',encoding='utf8')
        texts.append(f.read())
        print(filename)
        f.close()
        labels.append(0) # negative 값 라벨을 0 으로~

    posDir = folder_url/foldername/"pos"
    for filename in os.listdir(posDir):
        f = open(posDir/filename,'r',encoding='utf8')
        texts.append(f.read())
        print(filename)
        f.close()
        labels.append(1)

    return texts, labels
    
    # print(len(texts)) # 12500
    # print(len(labels)) # 12500

# train_data, train_labels = makeData("train")
# test_data, test_labels = makeData("test")

# 파이썬에서는 객체를 그대로 저장했다가 원복 시켜주는 라이브러리가 있다
import pickle
# 저장하기
def saveData():
    data = {"train_data":train_data, "train_labels":train_labels, "test_data":test_data,
            "test_labels":test_labels}
    f = open("imdb_data.bin","wb") # binary로 저장할 때는 보통 파일 확장자를 bin으로 사용
    pickle.dump(data,file=f)
    f.close()

def loadData():
    f = open("imdb_data.bin","rb")
    data2 = pickle.load(file=f)
    f.close()
    train_data = data2["train_data"]
    train_labels = data2["train_labels"]
    test_data = data2["test_data"]
    test_labels = data2["test_labels"]
    return train_data, train_labels, test_data, test_labels

# saveData()
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
max_len = 20 # 단어 20개만!
# 문장 길이 맞추기
data = pad_sequences(sequences, maxlen=max_len, truncating='post')

import numpy as np
def train_test_split(data, labels, test_size=0.2):
    #데이터를 섞어서 쪼개기 - train_test_split 안쓰고 
    rows_count = 200  #data.shape[0] #데이터 개수 저장하기(데이터가 너무 커서 200개만..)
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

embedding_dim = 512
model = Sequential()
# 임베딩 층에 먼저 넣기! 입력층, 출력층, 각 문장 길이!
model.add(Embedding(10000, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train,y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

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