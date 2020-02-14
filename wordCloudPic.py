# -*- coding: utf-8 -*-
"""

@author: Hogan
"""

import jieba
from wordcloud import WordCloud
import pandas as pd
import readFile
import matplotlib.pyplot as plt
import re

def jieba_cut(data):
    a = []
    worddict = {}
    for i in data:
        words = jieba.lcut(i)
        a.extend(words)
    for word in a:
        worddict.setdefault(word,0)
        worddict[word]+=1
    return worddict

def deal_with_meanless_word(data):
    mean_words = {}
    for i in data.keys():
        if len(i) > 1:#认为小于一个长度的字没有意义，最好的情况下是自己定义一个没有意义的列表
            mean_words[i] = data[i]
    return mean_words

