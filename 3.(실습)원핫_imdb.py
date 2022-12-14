import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras import models, layers
from keras.preprocessing.text import Tokenizer

# 뉴스 기사가 긍정(1)인지 부정(0)인지 분류하기!

#1. 데이터 가져오기
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
print(X_train.shape)
print(y_train.shape)
# print(X_train[0]) # [1, 14, 22, 16, 43, 530, ...  이렇게 시퀀스 배열로 되어있음

# 원본 파일 받아오기!!
# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz


#1. 파일을 읽고, 라벨링한다. 문자열은 문자열대로 덧붙이고, 라벨은 라벨대로
#2. 인덱싱화(시퀀스를 만들고)
#3. 원핫인코딩하고 학습하기
import os, shutil, pathlib

def makeData(foldername='train'):
    # 해당 데이터셋은 train, test가 하위폴더로 나뉘어 있으므로
    # 폴더이름만 train, test로 바꿔주면 데이터셋을 따로 저장할 수 있다
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

# train,test 데이터를 나누어 받기
# train_data, train_labels = makeData("train")
# test_data, test_labels = makeData("test")

# 받은 데이터를 피클 파일로 저장
# 파이썬에서는 객체를 그대로 저장했다가 원복 시켜주는 라이브러리가 있다
import pickle
# 저장하기
def saveData():
    data = {"train_data":train_data, "train_labels":train_labels, "test_data":test_data,
            "test_labels":test_labels}
    f = open("imdb_data.bin","wb") # binary로 저장할 때는 보통 파일 확장자를 bin으로 사용
    pickle.dump(data,file=f)
    f.close()

# 피클 데이터 로드하기
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

# 생성된 객체로 인덱스 목록 만들기 
# 아마 내부적으로 빈도수가 높은 것 순으로 들어갈듯 ^^..!
tokenizer.fit_on_texts(train_data)
print(tokenizer.word_index) # 어떻게 인덱싱 됐는지..
print(tokenizer.word_counts) # 각 단어별 개수는..?

print(len(tokenizer.word_index)) # 88582
print(len(tokenizer.word_counts)) # 88582

# text 원핫인코딩
# 1~10000까지의 단어 인덱스가 열이 되고, 그게 있으면 1 없으면 0 -> 1 0 0 1 0 0 0 1 0 ..
sequences = tokenizer.texts_to_matrix(train_data) # (25000, 10000)
print(sequences.shape)
print(sequences[:20])

# list -> numpy 배열로
import numpy as np
train_labels = np.array(train_labels)
model = keras.Sequential(
    [
        layers.Dense(64,activation='relu'),
        layers.Dense(32,activation='relu'),
        layers.Dense(16,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ]
)
model.build( input_shape=(None,10000))
print(model.summary())

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(sequences, train_labels, epochs=5)
