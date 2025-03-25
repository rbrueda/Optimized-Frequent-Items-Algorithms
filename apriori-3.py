import pandas as pd
import math
import mmh3
from bitarray import bitarray
from collections import defaultdict
from cuckoofilter import CuckooFilter  # Import Cuckoo Filter library

# Hybrid Bloom Filter + Cuckoo Hashing
class HybridFilter:
    def __init__(self, items_count, fp_prob):
        self.bloom_filter = BloomFilter(items_count, fp_prob)
        self.cuckoo_filter = CuckooFilter(capacity=items_count)
    
    def add(self, item):
        self.bloom_filter.add(item)
        self.cuckoo_filter.insert(item)
    
    def check(self, item):
        if self.bloom_filter.check(item):  # Bloom says it's present
            return self.cuckoo_filter.contains(item)  # Validate using Cuckoo
        return False

# Bloom Filter
class BloomFilter:
    def __init__(self, items_count, fp_prob):
        self.size = self.get_size(items_count, fp_prob)
        self.hash_count = self.get_hash_count(self.size, items_count)
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)
    
    def add(self, item):
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            self.bit_array[digest] = True
    
    def check(self, item):
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if not self.bit_array[digest]:
                return False
        return True
    
    @staticmethod
    def get_size(n, p):
        return int(-(n * math.log(p)) / (math.log(2) ** 2))
    
    @staticmethod
    def get_hash_count(m, n):
        return int((m / n) * math.log(2))

# Trie for storing frequent itemsets
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEndOfSet = False
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, itemset):
        current = self.root
        for item in sorted(itemset):
            if item not in current.children:
                current.children[item] = TrieNode()
            current = current.children[item]
        current.count += 1
        current.isEndOfSet = True
    
    def search(self, itemset):
        current = self.root
        for item in sorted(itemset):
            if item not in current.children:
                return False
            current = current.children[item]
        return current.isEndOfSet

# Vertical Data Representation for iApriori

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

# Frequent Itemset Mining using Hybrid Filtering
def getFrequentItemsets(vertical_df, min_support, hybrid_filter):
    freq_items = set()
    for item, tids in vertical_df.items():
        if len(tids) >= min_support:
            itemset = frozenset([item])
            hybrid_filter.add(str(itemset))
            freq_items.add(itemset)
    return freq_items

def joinSet(itemSet, length):
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length and i != j])

def iApriori(df, min_support, min_confidence):
    vertical_df, _ = getVerticalData(df)
    hybrid_filter = HybridFilter(10000, 0.3)
    trie = Trie()
    oneLSet = getFrequentItemsets(vertical_df, min_support, hybrid_filter)
    
    k = 2
    all_frequent_sets = {1: oneLSet}
    currentLSet = oneLSet

    while currentLSet:
        print(f"currentLSet {k}:")
        print(currentLSet)
        trie.insert(currentLSet)
        currentCSet = joinSet(currentLSet, k)
        newLSet = set()
        
        for candidate in currentCSet:
            tids = set.intersection(*[vertical_df[item] for item in candidate]) if all(item in vertical_df for item in candidate) else set()
            
            if len(tids) >= min_support and not hybrid_filter.check(str(candidate)):
                newLSet.add(candidate)
                hybrid_filter.add(str(candidate))
        
        if not newLSet:
            break
        
        all_frequent_sets[k] = newLSet
        currentLSet = newLSet
        k += 1
    
    return all_frequent_sets

# Load dataset and run iApriori
df = pd.read_csv('result2.csv')
ratings = df.drop(columns='user_id')
ratings_values = ratings.values.flatten()
mean_item = ratings_values.mean()
median_item = pd.Series(ratings_values).median()
std_item = pd.Series(ratings_values).std()
skewness = (3 * (mean_item - median_item)) / std_item

if skewness > 1.5:
    min_support = 0.1
    lift_threshold = 2.0
elif 0.5 <= skewness <= 1.5:
    min_support = 0.05
    lift_threshold = 1.3
else:
    min_support = 0.02
    lift_threshold = 1.0

support_threshold = int(min_support * len(df))
confidence_threshold = 0.6

frequent_itemsets = iApriori(df, support_threshold, confidence_threshold)

for k, count in {k: len(v) for k, v in frequent_itemsets.items()}.items():
    print(f"Number of {k}-itemsets: {count}")
#figure out why it is so slow....

#todo: figure out why it gets stuck after gemerating pairs of 3 (I think it may be generating all the pairs and not pruning....)
#stats: took  3 hours of running and still stuck