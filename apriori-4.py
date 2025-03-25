import numpy as np
from collections import defaultdict
from scipy.stats import skew

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.support = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, itemset, support):
        node = self.root
        for item in sorted(itemset):
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]
        node.is_end = True
        node.support = support
    
    def search_prefix(self, itemset):
        node = self.root
        for item in sorted(itemset):
            if item not in node.children:
                return None
            node = node.children[item]
        return node.support if node.is_end else None

def compute_dynamic_minsup(data):
    transaction_lens = [len(t) for t in data]
    skewness = skew(transaction_lens)
    if abs(skewness) > 1.5:
        return 0.1
    elif 0.5 <= abs(skewness) <= 1.5:
        return 0.05
    else:
        return 0.02

def apriori_tid(data, min_support, min_confidence, previous_knowledge=None):
    trie = Trie()
    vertical_db = defaultdict(set)
    for tid, transaction in enumerate(data):
        for item in transaction:
            vertical_db[frozenset([item])].add(tid)
    
    frequent_itemsets = {}
    k = 1
    while vertical_db:
        new_itemsets = {}
        for itemset, tids in vertical_db.items():
            support = len(tids) / len(data)
            if support >= min_support:
                trie.insert(itemset, support)
                frequent_itemsets[itemset] = support
                for other_itemset in frequent_itemsets:
                    if len(itemset | other_itemset) == k + 1:
                        new_tids = tids & vertical_db[other_itemset]
                        if new_tids:
                            new_itemsets[itemset | other_itemset] = new_tids
        vertical_db = new_itemsets
        k += 1
    
    return frequent_itemsets

def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        for item in itemset:
            antecedent = frozenset([item])
            consequent = itemset - antecedent
            if consequent:
                conf = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                lift = conf / frequent_itemsets[consequent] if frequent_itemsets[consequent] > 0 else 0
                if conf >= min_confidence and lift > 1:
                    rules.append((antecedent, consequent, conf, lift))
    return rules

def incremental_update(data, previous_frequent_itemsets, min_support):
    updated_data = previous_frequent_itemsets.copy()
    new_data = apriori_tid(data, min_support, 0.6, previous_frequent_itemsets)
    updated_data.update(new_data)
    return updated_data
 