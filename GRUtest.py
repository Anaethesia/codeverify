import os
import sys
import numpy as np
import keras
from keras import layers
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

whole = open('newd.txt', encoding='utf-8').read()

maxlen = 50  # 序列长度
sentences = []  # 存储提取的句子
next_chars = []  # 存储每个句子的下一个字符（即预测目标）

for i in range(0, len(whole) - maxlen):
    sentences.append(whole[i: i + maxlen])
    next_chars.append(whole[i + maxlen])
print('提取的句子总数:', len(sentences))

chars = sorted(list(set(whole))) # 语料中所有不重复的字符
char_indices = dict((char, chars.index(char)) for char in chars)
print('字典长度:',len(chars))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)  # 维张量（句子数，序列长度，字典长度）
y = np.zeros((len(sentences), len(chars)), dtype=np.bool) # 2维张量 （句子数，字典长度）
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1.0
    y[i, char_indices[next_chars[i]]] = 1.0

print(np.round((sys.getsizeof(x) / 1024 / 1024 / 1024), 2), "GB")

model=load_model('m4.h5')


def sample(preds, temperature=1.0):
    if not isinstance(temperature, float) and not isinstance(temperature, int):
        print('\n\n', "temperature must be a number")
        raise TypeError

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def write(model, temperature, word_num, begin_sentence):
    gg = begin_sentence[:50]  # 初始文本
    print(gg, end='\\\ ')
    for _ in range(word_num):
        sampled = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(gg):
            sampled[0, t, char_indices[char]] = 1.0

        preds = model.predict(sampled, verbose=0)[0]
        if temperature is None:  # 不加入temperature
            next_word = chars[np.argmax(preds)]
        else:
            next_index = sample(preds, temperature)  # 加入temperature后抽样
            next_word = chars[next_index]

        gg += next_word
        gg = gg[1:140]
        sys.stdout.write(next_word)
        sys.stdout.flush()

begin_sentence="女装业务的回暖（特别是上半年）以及医美新增并表推动18年前三季度公司营业收入和净利润分别同比增长21" "37%和8058%，投资收益贡献部分盈利增长"
print("初始句子：",begin_sentence[:50])
print("\n")
print("--------------------------------------------------")
write(model, 0.5, 150, begin_sentence)
print("\n")
print("--------------------------------------------------")
write(model, 1, 150, begin_sentence)
print("\n")
print("--------------------------------------------------")
write(model, 1.5, 150, begin_sentence)


# begin_sentence = whole[50003: 50100]
# print("初始句：", begin_sentence[:50])
#write(model, 0.5, 450, begin_sentence)
#write(model, 1, 450, begin_sentence)
