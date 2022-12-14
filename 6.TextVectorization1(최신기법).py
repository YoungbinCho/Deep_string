# Tokenize 만드는 원리
import string # 각종 특수문자들
print( string.punctuation)

# 시퀀스를 만드는 클래스 만들어보기
class Vectorize:
    def standarize(self,text):
        text = text.lower() # 전체 문장 소문자로 만들기
        return "".join(char for char in text\
                        if char not in string.punctuation) 
        # 특수문자 제거 후 다시 문장으로 만들어 내보내기

    def tokenize(self,text):
        return text.split() # 공백으로 문자열을 잘라 토큰으로 만들어서 반환

    def make_vocabulary(self, dataset):
        self.vocabulary = {"":0, "[UNK]":1} # 0, 1, 즉 앞에 두개는 사용 안 함
        for text in dataset:
            text = self.standarize(text)
            tokens = self.tokenize(text)

            for token in tokens:
                # 기존 vocabulary에 없는 단어라면 추가, 라벨은 길이로 붙임(그럼 새 숫자임!)
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary) 

        self.inverse_vocabulary = dict((v,k) for k, v in self.vocabulary.items())
        # school:3 => 3:school
        return self.vocabulary, self.inverse_vocabulary
    
    # vocaulary를 이용해서 시퀀스 만들기
    def encode(self, text):
        text = self.standarize(text) # 문장에서 각종 구두점을 뺀다
        tokens = self.tokenize(text) # 공백을 기준으로 단어를 쪼갠다.
        return [self.vocabulary.get(token, 1) for token in tokens]

    # 시퀀스를 다시 문장으로
    def decode(self, int_sequence):
        return " ".join(self.inverse_vocabulary.get(i,"[UNK]") for i in int_sequence)
        # get() : 딕셔너리의 키를 넣으면 값을 반환, 키에 대한 값이 없으면 [UNK]를 반환.

vectorize = Vectorize()
dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy boolms",
    "I hate test",
    "me too",
    "This is desk, That is chair",
    "He is a great hero"
]

v1,v2 = vectorize.make_vocabulary(dataset)
print(v1)
print(v2)

en = vectorize.encode("This is token test")
print(en)
de = vectorize.decode(en)
print(de)
