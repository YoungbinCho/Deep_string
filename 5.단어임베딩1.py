import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb # 영화리뷰 감성 분석 -> 긍정 1 부정 0
from keras.utils import pad_sequences

max_features = 10000 # 자주 사용하는 단어 10000개만 처리한다
max_len = 20 # 전체 문장을 사용할 수 없으므로 한 문장은 20개의 단어만 처리한다.
(X_train, y_train), (X_test, y_test) = imdb.load_data()

# 이미 시퀀스이다. 시퀀스로 만들어서 list에 담아온다
print(X_train[:3])

# 위 시퀀스들 중에서 단어 20개만 추출 또는 단어가 20개가 안 될 경우에는 20개를 강제로 만든다
# 앞/뒤 중 단어를 추출할 위치를 truncating으로 정할 수 있다.(truncating="pre":뒤에서 자름, 디폴트, truncating="post":앞에서 잘라옴)
X_train = pad_sequences(X_train, maxlen=max_len, truncating="post")
X_test = pad_sequences(X_test, maxlen=max_len, truncating="post")
print("padding 후")
print(X_train[:3])

# 밀집 공간 벡터 - Embedding 층 : 단어와 단어 사이의 거리를 계산하는 학습을 한다.

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
# Embedding : 제일 처음 오는 층
# max_features = 입력 차원
# 8 - 출력 차원
# input_length = 문장의 길이
# 이 층이 단어들의 관계를 계산한다
model.add( Embedding(max_features, 8, input_length=max_len)) # 출력이 3차원
model.add( Flatten()) # 완전 연결 레이어처럼 2차원으로 맞춰야
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss="binary_crossentropy", metrics=['acc'])
model.summary()

history  = model.fit(X_train,y_train, epochs=10, batch_size=32, validation_split=0.2)
# 자기가 알아서 0.2 만큼 잘라서 검증셋으로 사용한다.