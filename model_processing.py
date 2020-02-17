# -*- coding: utf-8 -*-
"""

@author: Hogan
"""

import readFile
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


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
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)
    data=[(X_train,Y_train),(X_test,Y_test)]    
    models = [('SVN',SVR(kernel='poly',C=10,degree=2)),('RFT',RandomForestRegressor(n_estimators=1000)),('adb',AdaBoostRegressor(learning_rate=0.01,n_estimators=1000))]
    for clf in models:
        clf_name, clf_param = clf
        clf_param.fit(X_train,Y_train)
        for i , db in enumerate(data):
            x, Y = db
            y_pred = clf_param.predict(x)
            print('%s' %('训练集' if i==0 else '测试集'),clf_name, metrics.mean_absolute_error(Y, y_pred))
        
if __name__ == '__main__':
    listing = readFile.listing
    data_processing_modeling(listing)