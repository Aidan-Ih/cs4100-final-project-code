import numpy as np
import pandas as pd
from bloom import BloomFilter

#define test parameters
DATA_PATH = 'malicious_url_scores.csv' 
filter_size = 300000
size_of_model = 227200 

data = pd.read_csv(DATA_PATH)
data.loc[data['label'] == 'malicious', 'label'] = 1
data.loc[data['label'] == 'benign', 'label'] = 0

negative_sample = data.loc[(data['label'] == 0)]
positive_sample = data.loc[(data['label'] == 1)]

url = positive_sample['key']
n = len(url)
    
#optimize BF and threshold T
def find_optimal_params(max_thresh, min_thresh, size, neg_sample):
    fp_opt = neg_sample.shape[0]
    
    for threshold in np.arange(.5, .99, 0.01):
        pos_keys = positive_sample.loc[(positive_sample['score'] <= threshold),'key']
        n = len(pos_keys)
        bloom_filter = BloomFilter(n, size)
        for key in pos_keys:
            bloom_filter.insert(key)
            
        ml_pos = neg_sample.loc[(neg_sample['score'] > threshold),'key']
        bloom_negative = neg_sample.loc[(neg_sample['score'] <= threshold),'key']
        bf_pos = 0
        for i in bloom_negative:
            bf_pos += bloom_filter.test(i)
        fp_total = bf_pos + len(ml_pos)
        if fp_total < fp_opt:
            fp_opt = fp_total
            thresh_opt = threshold
            bf_opt = bloom_filter
            
    return thresh_opt, bf_opt


        
T, bloom_filter = find_optimal_params(.95, .5, filter_size - size_of_model, negative_sample.sample(frac = 0.3))
    
count = 0
for index, row in negative_sample.iterrows():
    #first query ML
    if row['score'] > T:
        count += 1
    else:
        count += bloom_filter.test(row['key'])
    
print('False positive items: ', count)
print('FPR: ', count / len(negative_sample))