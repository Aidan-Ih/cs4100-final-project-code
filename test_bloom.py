import numpy as np
import pandas as pd
from bloom import BloomFilter

#define test parameters
DATA_PATH = 'malicious_url_scores.csv' 
filter_size = 250000 #bits

data = pd.read_csv(DATA_PATH)
data.loc[data['label'] == 'malicious', 'label'] = 1
data.loc[data['label'] == 'benign', 'label'] = 0

negative_sample = data.loc[(data['label'] == 0)]
positive_sample = data.loc[(data['label'] == 1)]

url = positive_sample['key']
n = len(url)

#setup bloom filter
bloom_filter = BloomFilter(n, filter_size)

#insert positive elements into bloom filter
for index, row in positive_sample.iterrows():
    bloom_filter.insert(row['key'])
    
#query all negative elements to calculate FPR
count = 0
for index, row in negative_sample.iterrows():
    count += bloom_filter.test(row['key'])
    
print('False positive items: ', count)
print('FPR: ', count / len(negative_sample))