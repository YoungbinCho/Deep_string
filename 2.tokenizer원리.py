import numpy as np
samples = [
    "The cat sat on the mat",
    "The dog ate my homework",
    "The cat is very cute",
    "THe fox is very smart and very beautiful and follow"
]

token_index = {} # 단어 사전임,
# {1:"the", 2:"cat",}

# 문장 하나만 token으로 나누자(단어 단위로)
s = samples[0]
sList = s.split(" ")
print(sList)
# token_index[sList[0]] = 1
# token_index[sList[2]] = 2
# token_index[sList[3]] = 3
# print(token_index) # {'The': 1, 'sat': 2, 'on': 3}

for word in samples[0].split():
    if word not in token_index:
        token_index[word] = len(token_index)+1
        # 뒤에 추가 된 것은 뒷 번호가 붙어야 하므로 개수에 +1 더해줌 ^^ 천재신가요?
print(token_index)

