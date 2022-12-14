# TextVectorization 객체 사용법(최신 툴)

import re
import string
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# 콜백 함수 - TextVectorization한테 전달가능, 일부는 내가 만든걸로

# 기초적인 전처리 함수 - 소문자화, 특수문자 제거
def custom_standarization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor) # 문장 받아서 소문자로 바꾸기
    return tf.strings.regex_replace(
        lowercase_string, f"[{re.escape(string.punctuation)}]",""
    ) # 특수문자 제거

# 토큰화 함수
def custom_split_fn(string_tensor): 
    return tf.strings.split( string_tensor)

# TextVectorization 객체에 위의 함수들을 매개변수로 전달(전처리 함수를 직접 만들 수 있음)
text_vectorization = TextVectorization(
    output_mode = "int", # 단순 숫자 시퀀스로 만들긴
    standardize = custom_standarization_fn,
    split = custom_split_fn
)

dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A puppy boolms",
    "I hate test",
    "me too",
    "This is desk, That is chair",
    "He is a great hero"
]

text_vectorization.adapt(dataset) # 데이터 벡터화 작업을 진행한다.
print(text_vectorization.get_vocabulary()) # 토큰 사전 가져오기
s = "I hate puppy"
print( text_vectorization(s)) # encode, s의 시퀀스 가져오기

inverse_vocab = dict(enumerate(text_vectorization.get_vocabulary()))
print( inverse_vocab)
encode = text_vectorization(s)
decode = " ".join(inverse_vocab[int(i)] for i in encode) # 시퀀스를 가져가 해당 단어 가져오기
print(decode)

# aclImdb로 실습
# text_dataset_from_directory : 디렉토리에서 파일을 직접 읽어온다

from tensorflow import keras
batch_size=32 # 배치 - 한꺼번에 처리할 사이즈 32로 지정

train_ds = keras.utils.text_dataset_from_directory(
    "./datasets/aclImdb/train", batch_size=batch_size
)

test_ds = keras.utils.text_dataset_from_directory(
    "./datasets/aclImdb/test", batch_size=batch_size
)
 
for inputs, targets in train_ds:
    print(inputs.shape)
    print(inputs.dtype)
    print(targets.shape)
    print(targets.dtype)
    print(inputs[0])
    print(targets)
    break # 32개만 가져오기 

# tf-idf 가 텐서 2.8 gpu에서 오류, 
text_vectorization = TextVectorization(
    max_tokens = 20000,
    output_mode = "multi_hot" # 원핫 인코딩
)

# 벡터라이징 할 x 값만 가져오기
text_only_train_ds = train_ds.map(lambda x,y: x) # 라벨 떼고 텍스트만
# 데이터셋에 대한 단어사전 만들기!!
text_vectorization.adapt(text_only_train_ds)

for i in text_only_train_ds:
    print(i)
    break
# gpu장치를 안 쓰고 cpu로 작업하고자 할 때
# with tf.device("cpu"):
#   text_vectorization.adapt(text_only_train_ds)

# 벡터화시킬 문자열인 x만 객체에 넣어줌
# map(num_parallel_calls) 를 하면 여러개의 스레드로 나누어서 속도를 높여 처리 가능
# train_ds가 tf 함수를 통해 받아온 데이터여서 tf의 map함수가 적용된 듯 함. 그래서 저런 매개변수 사용 가능
binary_1gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
)

binary_1gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
)

from tensorflow.keras import layers
def getModel(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x) # 과적합 심할 때 절반 쯤 버린다
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs,outputs)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = getModel()
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("binary_1gram.keras", save_best_only=True)
]

model.fit(binary_1gram_train_ds.cache(),
            validation_data = binary_1gram_test_ds.cache(),
            epochs=10,
            callbacks=callbacks)

# 끝나면 저장되어 있으니까 모델 불러오기
model = keras.models.load_model('binary_1gram.keras')
print(model.evaluate(binary_1gram_test_ds)[1])