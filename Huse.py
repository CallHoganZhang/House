# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:39:45 2019

@author: Administrator
"""

from matplotlib import font_manager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import scipy.stats as ss
#import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler,StandardScaler,Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis#LDA降维
from sklearn import metrics
import warnings
from keras.models import Sequential
from keras.layers.core import Dense,Activation
warnings.filterwarnings("ignore")


listing_path = 'listings.csv'
neighbourhoods_path = 'neighbourhoods.csv'
listing = pd.read_csv(listing_path,encoding='utf-8')
neighbourhoods = pd.read_csv(neighbourhoods_path)
print(listing.head())
print(neighbourhoods.head())
listing.info()
listing['neighbourhood'].sample(10)
a = listing['neighbourhood']
def colum_to_str(data):
    neighbourhood = []
    a = data.str.findall('\w+').tolist()
    for i in a:
        neighbourhood.append(i[0])
    return neighbourhood

listing['neighbourhood'] = colum_to_str(a)
print(listing['neighbourhood'].unique())

print(listing['room_type'].unique())
new_columns = ['price','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']
data = listing[new_columns]


def is_number(data):
    int_columns = []
    str_columns = []
    columns_name = data.columns.tolist()
    for i in range(data.shape[1]):
        if data[columns_name[i]].dtype == 'int64' or data[columns_name[i]].dtype == 'float64':
            int_columns.append(columns_name[i])
        else:
            str_columns.append(columns_name[i])
    return int_columns, str_columns   

int_columns, str_columns = is_number(listing)

for i in range(len(data.columns)):
    plt.hist(data[data.columns[i]].get_values())
    plt.xlabel(data.columns[i])
    plt.savefig('./' + data.columns[i]+'.png')
    plt.show()
    
    
data.describe()
plt.subplot(311)
data = listing['room_type'].value_counts().tolist()
a = listing['room_type'].unique()
print('room_type', a)

plt.bar([0,1,2], data)
plt.show()



listing[['host_name','name']].groupby('host_name').count().sort_values(by='name', ascending=False).head()
listing['neighbourhood'].value_counts()
#
def explode_situtation(data):
    explode = {}
    for i in range(len(data)):
        if data[i]>data.mean():
            explode[data.index[i]] = 0.1
        else:
            explode[data.index[i]] = 0
    return explode

explode = list(explode_situtation(listing.neighbourhood.value_counts()).values())
print(explode)

data2 = listing.neighbourhood.value_counts()
label2 = listing.neighbourhood.unique().tolist()

plt.figure(figsize=(12,12))
plt.title('民宿区域分布比例图',fontdict={'fontsize':18})
plt.pie(data2,labels=label2,autopct='%.2f%%',explode=explode,startangle=90,
        counterclock=False,textprops={'fontsize':12,'color':'black'})
plt.legend(loc='best',shadow=True,fontsize=11)
plt.savefig('./distrubte.png')

a = listing[['neighbourhood','price']].groupby(['neighbourhood','price']).count().reset_index()
for i in label2:
    plt.hist(a[a['neighbourhood']==i].price)
    plt.xlabel(i)
    plt.savefig('./'+str(i)+'.png')
    plt.show()
    
price_is_0 = listing[listing['price']==0] #exception
print(price_is_0)


test_house = listing[listing.name.str.startswith('测试')==True]
print(test_house)

drop_index_list = price_is_0.index.tolist()+test_house.index.tolist()
listing_dealt = listing.drop(drop_index_list)
listing_dealt[listing_dealt['price']==0]

listing_dealt.head(3)

avg_review = listing_dealt['number_of_reviews'].quantile(0.9)
print('avg_review', avg_review)
#


import jieba
from wordcloud import WordCloud

reviews_top90 = listing_dealt.sort_values(by=['number_of_reviews'],ascending=False)
print(reviews_top90.head())
#
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
popular_words = jieba_cut(reviews_top90.name.astype('str'))

print(popular_words)
#
plt.figure(figsize=(15,8))
plt.title('最受欢迎的房间中描述关键词')
#
def deal_with_meanless_word(data):
    mean_words = {}
    for i in data.keys():
        if len(i) > 1:#认为小于一个长度的字没有意义，最好的情况下是自己定义一个没有意义的列表
            mean_words[i] = data[i]
    return mean_words

mean_words = deal_with_meanless_word(popular_words)
mean_words_df = pd.Series(mean_words).sort_values(ascending=False)
mean_words_df_top15 = mean_words_df.head(15)
print(mean_words_df_top15)
mean_words_df_top15.plot(kind='bar')
import re
wordcloud_use = ' '.join(mean_words.keys())
resultword=re.sub("[A-Za-z0-9]", "",wordcloud_use) 
print('resultword', resultword)
#
w = WordCloud(scale=4,background_color='white', font_path='SIMLI.TTF', 
             max_words = 2000,max_font_size = 20,random_state=20).generate(resultword[:200])
w.to_file('result.jpg')


def get_con(df):
    subsets=['price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']
    data={}
    for i in subsets:
        data.setdefault(i,[])
        data[i].append(df[i].skew())
        data[i].append(df[i].kurt())
        data[i].append(df[i].mean())
        data[i].append(df[i].std())
        data[i].append(df[i].std()/df[i].mean())
        data[i].append(df[i].max()-df[i].min())
        data[i].append(df[i].quantile(0.75)-df[i].quantile(0.25))  #分位数
        data[i].append(df[i].median())
    data_df=pd.DataFrame(data,index=['偏度','峰度','均值','标准差','变异系数','极差','四分位距','中位数'],columns=subsets)
    return data_df.T

df2=get_con(listing)
df2.to_csv('eval.csv')


def data_processing_modeling(df):
    model_df = df.drop('neighbourhood_group',axis = 1)
    model_df = df.dropna(how='any',subset=['reviews_per_month','last_review'])
    model_df['name_length']=model_df.name.apply(lambda x:len(str(x).split()))
    model_df['host_name_length']=model_df.host_name.apply(lambda x:len(str(x)))
    model_df['neighbourhood_digit']=LabelEncoder().fit_transform(model_df.neighbourhood)
    model_df['room_digit']=LabelEncoder().fit_transform(model_df.room_type)
    feature_subsets=['name_length','host_name_length','neighbourhood_digit','room_digit','price','minimum_nights']
    label=['reviews_per_month']
    for i in feature_subsets:
        model_df[i]=MinMaxScaler().fit_transform(model_df[i].values.reshape(-1,1)).reshape(1,-1)[0]
    X = model_df[feature_subsets]
    y = model_df[label]
    NN = Sequential()
    NN.add(Dense(len(feature_subsets)))
    NN.add(Activation('relu'))
    NN.add(Dense(1))
    NN.add(Activation('softmax'))
    NN.compile(optimizer='sgd', loss = 'mse')
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)
    data=[(X_train,Y_train),(X_test,Y_test)]    
    models = [('SVN',SVR(kernel='poly',C=10,degree=2)),('RFT',RandomForestRegressor(n_estimators=1000)),('adb',AdaBoostRegressor(learning_rate=0.01,n_estimators=1000))]
    #('NN',NN)
    
    for clf in models:
        clf_name, clf_param = clf
        clf_param.fit(X_train,Y_train)
        for i , db in enumerate(data):
            x, Y = db
            y_pred = clf_param.predict(x)
            print('%s' %('训练集' if i==0 else '测试集'),clf_name, metrics.mean_absolute_error(Y, y_pred))
        
    
listing = pd.read_csv(listing_path)   
    
data_processing_modeling(listing)
    






















