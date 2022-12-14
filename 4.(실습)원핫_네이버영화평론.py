# 네이버 영화 평론이 긍정인지 부정인지 나누기
import tensorflow as tf 
from tensorflow import keras 
from keras.preprocessing.text import Tokenizer  #토큰으로 만들어 주는 라이브러리 
from keras.datasets import imdb
from keras import models, layers
from keras.preprocessing.text import Tokenizer  
import pandas as pd 

train_data = pd.read_table("./datasets/ratings.txt")
print( train_data[:5]) # id, document(후기), label 로 구성
print( type(train_data)) # 데이터프레임 형태

# NaN데이터 먼저 삭제 
train_data = train_data.dropna(axis=0) # 행 중에 NaN 들어가는것 삭제 
print( train_data[:5])

# 라벨나누고 
train_labels =train_data.iloc[:,2] # 맨 마지막 열이 라벨
print(train_labels[:5]) # 판다스

# id삭제하고  
train_data = train_data.iloc[:, 1:2]  #0 번열 삭제하기 
print( train_data[:5])

# dataframe -> list로 바꿔야 한다
# 그래야 fit_on_texts에 들어감  
texts =[]
for line in train_data["document"]:
    texts.append(line)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts( texts)
train_data = tokenizer.texts_to_matrix(texts) #ndarray로 바꾸었음
print(train_data.shape)
print( train_data[:5, :20])


import numpy as np 
train_labels = np.array(train_labels) # dataframe -> ndarray로 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, 
    test_size=0.3, random_state=1)

# 데이터가 너무 많아 3만개만 사용!
X_train2 = X_train[:30000]
y_train2 = y_train[:30000]
X_val = X_train[30000:40000]
y_val = y_train[30000:40000]

print( y_train[:20]) # 잘 섞였는지 확인

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
model.fit(X_train2, y_train2, epochs=5, validation_data=(X_test,y_test), batch_size=16)

