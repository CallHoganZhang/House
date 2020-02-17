# -*- coding: utf-8 -*-
"""

@author: Hogan
"""

import pandas as pd

listing_path = 'listings.csv'
neighbourhoods_path = 'neighbourhoods.csv'
listing = pd.read_csv(listing_path,encoding='utf-8')
neighbourhoods = pd.read_csv(neighbourhoods_path)

print("listing information")
print(listing.info())
print("listing describe")
print(listing.describe())

new_columns = ['price','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']


price_is_0 = listing[listing['price']==0]  
test_house = listing[listing.name.str.startswith('测试')==True]
drop_index_list = price_is_0.index.tolist() + test_house.index.tolist()
listing_dealt = listing.drop(drop_index_list)
listing_dealt[listing_dealt['price']==0]

print("listing_dealt head information")
print(listing_dealt.head(3))

avg_review = listing_dealt['number_of_reviews'].quantile(0.9)

reviews_top90 = listing_dealt.sort_values(by=['number_of_reviews'],ascending=False)
print("reviews_top90 head information")
print(reviews_top90.head())
