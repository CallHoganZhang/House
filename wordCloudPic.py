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

popular_words = jieba_cut(readFile.reviews_top90.name.astype('str'))
mean_words = deal_with_meanless_word(popular_words)
mean_words_df = pd.Series(mean_words).sort_values(ascending=False)
mean_words_df_top15 = mean_words_df.head(15)
print(mean_words_df_top15)
#plt.figure(figsize=(15,8))
#plt.title('最受欢迎的房间中描述关键词')
mean_words_df_top15.plot(kind='bar')

wordcloud_use = ' '.join(mean_words.keys())
resultword=re.sub("[A-Za-z0-9]", "",wordcloud_use) 

w = WordCloud(scale=4,background_color='white', font_path='SIMLI.TTF', 
             max_words = 2000,max_font_size = 20,random_state=20).generate(resultword[:200])
w.to_file('result.jpg')

