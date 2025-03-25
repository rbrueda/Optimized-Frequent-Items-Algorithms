#iapriori implementation
#bloom filters
#trie implementation

import pandas as pd
import scipy
from collections import defaultdict
from itertools import  chain, combinations
from optparse import OptionParser

from bitarray import bitarray
import math
import mmh3

import time

#bloom filter for itemset existence tracking
#test whether an element is a member of a set
#bloom filter -> space-efficent probabilistic data structure. Can be used to test whether an element is a member of a set
#implementation from: https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/ 
#store key instead of hash values
class BloomFilter:
    def __init__(self, items_count, fp_prob):       
        self.size = self.get_size(items_count, fp_prob) #size of bit array to use
        if self.size <= 0:
                raise ValueError("Bloom Filter size must be greater than 0.")


        self.fp_prob = fp_prob #false possible probability in decimal

        #number of hash functions to use
        self.hash_count = self.get_hash_count(self.size, items_count)

        #bit array of a given size
        self.bit_array = bitarray(self.size)

        #initialize all bits as 0
        self.bit_array.setall(0)

    #add an item in the filter
    def add(self, item):
        digests = []
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            digests.append(digest)

            #set the bit True in bit_array
            self.bit_array[digest] = True
    
    #check the exisitence of an item in filter    
    def check(self, item):
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if self.bit_array[digest] == False:
                return False
        return True

    @classmethod
    #return the size of bit array(m) to be used
    def get_size(self, n, p):
        m = -(n*math.log(p))/(math.log(2)**2)
        return int(m)
    
    @classmethod
    def get_hash_count(self, m, n):
        k = (m/n) * math.log(2)
        return int(k)

#code from: https://wangyy395.medium.com/implement-a-trie-in-python-e8dd5c5fde3a 
#Trie data structure -> consists of nodes connected by edges
#each node -> represents a part of list
#we will be using this to store frequent itemsets
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEndOfSet = False
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    #inserts a word into the trie
    def insert(self, itemset) -> None:
        current = self.root
        #tertes through each value in itemset
        #todo: figure out why sorted -> increases time complexity
        for item in sorted(itemset): 
            if item not in current.children:
                current.children[item]= TrieNode()
            current = current.children[item]
        current.count += 1 #trck the number of insertions to the trie
        current.isEndOfSet = True

    #return if the item is the trie
    def search(self, itemset) -> bool:
        current = self.root
        for item in sorted(itemset):
            if item not in current.children:
                return False
            current = current.children[item]
        return current.isEndOfSet
    
#convert dataset into vertical format -> Apriori TID
#vertical_df -> a dictionary that stores transactions in vertical format
#   - each key (item) maps to the set of transactions (TIDs) that contain the item
def getVerticalData(df):
    df = df.drop('user_id', axis=1)

    review_map = defaultdict(set)
    reviewsList = []

    for index, row in df.iterrows():
        reviews = frozenset(row.index[row == 1])
        reviewsList.append(reviews)
        for item in reviews:
            review_map[item].add(index)

    return review_map, reviewsList

#generate frequent itemsets using vertical representation
def getFrequentItemsets(vertical_df, ratingsList, min_support, bloom_filter):
    freq_items = set()
    for item, tids in vertical_df.items():
        support = len(tids)
        #check if the element is greater than or equal to min support
        if support >= min_support:
            itemset = frozenset([item])
            freq_items.add(itemset)
            bloom_filter.add(str(itemset))
    return freq_items

#returns a n-element itemset
def joinSet(itemSet, length):
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length and i != j]
    )

def iApriori(df, min_support, min_confidence):
    #convert data into vertical format
    vertical_df, ratingsList = getVerticalData(df)

    #apply a bloom filter on the data in order to efficient check if an element is a member of a set
    bloom_filter = BloomFilter(10000, 0.3)

    trie = Trie()

    #extracts frequent 1-itemsets -> updates freqSet with support counts
    oneLSet = getFrequentItemsets(vertical_df, ratingsList, min_support, bloom_filter)

    #set k value to start generating pairs for items
    k = 2
    all_frequent_sets = {1: oneLSet}

    #current prunned itemset 
    currentLSet = oneLSet

    while currentLSet:
        #store frequent itemset -> use trie for efficiency
        trie.insert(currentLSet)
        
        #generate candidate k-itemsets 
        currentCSet = joinSet(currentLSet, k)

        #set a new frequent itemset
        newLSet = set()

        for candidate in currentCSet:
            #efficinet support counting using vertical data representation from iApriori

            tids = set.intersection(*[vertical_df[item] for item in candidate]) if all(item in vertical_df for item in candidate) else set()
            if len(tids) >= min_support and not bloom_filter.check(str(candidate)):
                #add the new candidate item to frequent itemset
                newLSet.add(candidate)
                bloom_filter.add(str(candidate))

        if not newLSet:
            break #not more frequent itemsets to generate

        all_frequent_sets[k] = newLSet
        currentLSet = newLSet #set the frequent frequent itemsets to the one recently generated
        k += 1 # find k+1-grouping itemsets

    return all_frequent_sets

df = pd.read_csv('result2.csv')

# Step 1: Calculate Mean, Median, and Standard Deviation for the ratings (ignoring user_id)
ratings = df.drop(columns='user_id')  # Drop 'user_id' column to focus on ratings

# Flatten the DataFrame to get all the ratings as a single list
ratings_values = ratings.values.flatten()

# Calculate the mean, median, and standard deviation for all ratings
mean_item = ratings_values.mean()
median_item = pd.Series(ratings_values).median()
std_item = pd.Series(ratings_values).std()

skewness = (3 * (mean_item - median_item)) / std_item

#calculate minsupport using skewness
if skewness > 1.5:
    min_support = 0.1
    #set lift threshold based on skewness
    lift_threshold = 2.0 #stronger positive correlation - items occur together twice as often as expected
elif skewness <= 1.5 and skewness >= 0.5: 
    min_support = 0.05
    lift_threshold = 1.3 #mild positive association between items 
else:
    min_support = 0.02
    lift_threshold = 1.0 #two items are idenpendent of one another - occurence of one item does not affect the probability of the other item

support_threshold = int(min_support * len(df))
print(f"support threshold: {support_threshold}")
print(f"lift threshold: {lift_threshold}")
confidence_threshold = 0.6

start_time = time.time()

frequent_itemsets = iApriori(df, support_threshold, confidence_threshold)

# Dictionary to store the count of itemsets per k
itemset_counts = {k: len(v) for k, v in frequent_itemsets.items()}

# Print the results
for k, count in itemset_counts.items():
    print(f"Number of {k}-itemsets: {count}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

#figure out false positive rate in this... 
f = open("frequent-itemsets-result-2.txt", "w")

f.write(str(frequent_itemsets))

#get buckoo cuckoo hashing to reduce avoid false positives