#TextVectorization으로 임베딩 층 사용하기 
import re 
import string
import tensorflow as tf 
from tensorflow.keras.layers import TextVectorization, Embedding

#TextVectorization 

#콜백함수 - TextVectorization 한테 전달가능, 일부는 내가 만들어서 완성 가능 
def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower( string_tensor) #문장받아서 소문자로 바꾸기
    return tf.strings.regex_replace( 
        lowercase_string, f"[{re.escape(string.punctuation)}]", ""
    )#특수문자 제거 

def custom_split_fn(string_tensor): #토큰 나누기 
    return tf.strings.split( string_tensor)

#콜백함수를 반드시 줘야하는게 아니다. !토큰처리방식이 다른 언어들의 경우! 만들어서 줘야 한다 
#영어는 굳이 콜백 안 줘도 된다. 
max_length = 600
max_tokens = 20000
text_vectorization = TextVectorization( 
    output_mode="int",
    max_tokens=max_tokens,
    output_sequence_length=max_length  
)

#text_dataset_from_directory : 디렉토리에서 파일을 직접 읽어온다 

from tensorflow import keras 
batch_size=32 #배치 - 한꺼번에 처리할 사이즈 32로 지정 

train_ds = keras.utils.text_dataset_from_directory( 
    "./datasets/aclImdb/train", batch_size=batch_size
)

test_ds = keras.utils.text_dataset_from_directory( 
    "./datasets/aclImdb/test", batch_size=batch_size
)

#둘이 합하기 , map함수는 연산을 가하고 그 연산의 결과를 반환한다 
text_only_train_ds = train_ds.map(lambda x, y:x) #문자열만 가져오고 y는 필요없음
text_vectorization.adapt( text_only_train_ds )
print(text_only_train_ds)


binary_1gram_train_ds = train_ds.map( 
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
)

binary_1gram_test_ds = test_ds.map( 
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
)

from tensorflow.keras import layers
# 일반적인 모델
def getModel1(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x) #과대적합심할때 절반쯤 버린다
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])

    return model 

# 원핫인코딩 -> 양방향 순환신경망
# 순환신경망(RNN - recursive) : 기존의 신경망이 각 계층별로 메모리가 없어서 기억이 저장되지 않는다. 우리가 책을 읽으면서 앞의 내용을 기억하는 것 과는 다르게...
# 기억 소자를 붙여서 출력 -> 다음과정의 입력으로 사용된다
# 신경망 내부에 while문의 형태이다. 텍스트 처리와 시계열 자료 처리에 적합하다.
# CNN - 이미지 처리에 적합
# RNN - 시퀀스 처리, 시계열 자료 처리에 적합한 구조이다.
def getModel2(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(None,), dtype='int64')
    # 순환 신경망에서 쓰기 위해서 
    embedded = tf.one_hot(inputs, depth=max_tokens)
    print(embedded.shape)
    print(embedded[:2])

    x = layers.Bidirectional(layers.LSTM(32))(embedded) # LSTM - 순환신경망
    x = layers.Dropout(0.5)(x) #과대적합심할때 절반쯤 버린다
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])

    return model 

def getModel3(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(None,), dtype='int64')
    # 순환 신경망에서 쓰기 위해서 
    # input_dim : 들어가는 숫자 중 최대 숫자, output_dim: 결과값이 몇개로 나올지
    embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
    x = layers.Bidirectional(layers.LSTM(32))(embedded) # LSTM - 순환신경망
    x = layers.Dropout(0.5)(x) #과대적합심할때 절반쯤 버린다
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])

    return model 

model = getModel3(max_length) # shape는 max_length랑 맞춰줘야됨..(왜?)
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("binary_1gram.keras", save_best_only=True)
]

model.fit( binary_1gram_train_ds.cache(), 
           validation_data=binary_1gram_test_ds.cache(),
           epochs=10,
           callbacks=callbacks)
# 끝나면 저장되어 있으니까 모델 불러오기 
model = keras.models.load_model("binary_1gram.keras")
print(model.evaluate(binary_1gram_test_ds)[1])




