# -*- coding: utf-8 -*-
"""

@author: Hogan
"""

import pandas as pd

listing_path = 'listings.csv'
neighbourhoods_path = 'neighbourhoods.csv'
listing = pd.read_csv(listing_path,encoding='utf-8')
neighbourhoods = pd.read_csv(neighbourhoods_path)

new_columns = ['price','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']

