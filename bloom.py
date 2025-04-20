from mmh3 import hash as mmh3_hash
from random import randint
import numpy as np

def new_hashfunc(m):
    def h(x):
        return mmh3_hash(x, randint(0, 10000000)) % m
    return h

'''
Class for a basic bloom filter
Space/time trade-offs in hash coding with allowable errors, Burton H Bloom. https://dl.acm.org/doi/10.1145/362686.362692
'''
class BloomFilter():
    def __init__(self, n, hash_len):
        '''
        n: number of expected elements inserted into the filter
        hash_len: number of bits each element is hashed to (length of hash table)
        k: number of hash functions, as calculated by k = hash_len/n * ln(2)
        h: array of hash functions
        table: hash table
        '''
        self.n = n
        self.hash_len = hash_len
        if self.n > 0 and self.hash_len > 0:
            self.k = max(1,int(self.hash_len/n*0.693)) 
        elif self.n == 0:
            self.k = 1
        self.h = [new_hashfunc(self.hash_len) for i in range(self.k)]
        self.table = np.zeros(self.hash_len, dtype=int)
        
    def insert(self, key):
        '''
        Insert one key into the bloom filter
        '''
        if self.hash_len == 0:
            raise Exception('Cannot insert into hash table of size 0')
        for j in range(self.k):
            hash_index = self.h[j](key)
            self.table[hash_index] = 1

    def test(self, key):
        '''
        Test a single key in the bloom filter
        '''
        test_result = 0
        match = 0
        if self.hash_len > 0:
            for j in range(self.k):
                hash_index = self.h[j](key)
                match += self.table[hash_index]
            #if all hits, element is in the filter
            if match == self.k:
                test_result = 1
        return test_result