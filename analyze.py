# -*- coding: utf-8 -*-
"""

@author: Hogan
"""

import readFile
import matplotlib.pyplot as plt
import pandas as pd

def colum_to_str(data):
    neighbourhood = []
    a = data.str.findall('\w+').tolist()
    for i in a:
        neighbourhood.append(i[0])
    return neighbourhood

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
    data_df =  data_df.T
    data_df.to_csv('eval.csv')

def plotIndex(data):
    for i in range(len(data.columns)):
        plt.hist(data[data.columns[i]].get_values())
        plt.xlabel(data.columns[i])
        plt.savefig('./' + data.columns[i]+'.png')
        plt.show()

def show_explode_situtation(data):
    explode = {}
    for i in range(len(data)):
        if data[i]>data.mean():
            explode[data.index[i]] = 0.1
        else:
            explode[data.index[i]] = 0
    return explode

def  plot_rigion_proportion(data,explode):

    data2 = data.neighbourhood.value_counts()
    label2 = data.neighbourhood.unique().tolist()

    plt.figure(figsize=(12,12))
    plt.title('民宿区域分布比例图',fontdict={'fontsize':18})
    plt.pie(data2,labels=label2,autopct='%.2f%%',explode=explode,startangle=90,
            counterclock=False,textprops={'fontsize':12,'color':'black'})
    plt.legend(loc='best',shadow=True,fontsize=11)
    plt.savefig('./distrubte.png')

def showRoomStyle(data):
    data = data['room_type'].value_counts().tolist()
    room_type = data['room_type'].unique()
    print('room_type :', room_type)
    plt.bar([0,1,2], data)
    plt.show()


if __name__ == '__main__':

#    int_columns, str_columns = is_number(readFile.listing)
#    plotIndex(readFile.listing[readFile.new_columns])
        
#    explode = list(show_explode_situtation(readFile.listing.neighbourhood.value_counts()).values())
#    plot_rigion_proportion(readFile.listing, explode)
    
#    df2 = get_con(readFile.listing)
    
#    showRoomStyle(readFile.listing)
