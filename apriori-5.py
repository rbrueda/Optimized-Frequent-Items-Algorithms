import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import skew
from pybloom_live import BloomFilter

def load_data(file_path):
    df = pd.read_csv(file_path)
    ratings = df.drop(columns='user_id')
    return ratings

def calculate_skewness(ratings_values):
    mean_item = ratings_values.mean()
    median_item = pd.Series(ratings_values).median()
    std_item = pd.Series(ratings_values).std()
    return (3 * (mean_item - median_item)) / std_item

def determine_thresholds(skewness, df_length):
    if skewness > 1.5:
        min_support = 0.1
        lift_threshold = 2.0
    elif 0.5 <= skewness <= 1.5:
        min_support = 0.05
        lift_threshold = 1.3
    else:
        min_support = 0.02
        lift_threshold = 1.0
    
    support_threshold = int(min_support * df_length)
    confidence_threshold = 0.6
    return support_threshold, confidence_threshold, lift_threshold

def convert_to_transactions(ratings):
    transactions = []
    for _, row in ratings.iterrows():
        transaction = set(row[row > 0].index.astype(str))  # Only include items rated > 0
        transactions.append(transaction)
    return transactions

def generate_frequent_1_itemsets(transactions, support_threshold, bloom_filter):
    item_counts = defaultdict(int)
    
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
            bloom_filter.add(item)  

    freq_items = {frozenset([item]) for item, count in item_counts.items() if count >= support_threshold}

    return freq_items

def generate_candidates(prev_itemsets, length):
    candidates = set()
    prev_list = list(prev_itemsets)

    for i in range(len(prev_list)):
        for j in range(i + 1, len(prev_list)):
            union_set = prev_list[i] | prev_list[j]
            if len(union_set) == length:
                subsets = {frozenset(union_set - {x}) for x in union_set}
                if subsets.issubset(prev_itemsets):
                    candidates.add(frozenset(union_set))

    return candidates

def mine_frequent_itemsets(transactions, support_threshold, freq_itemsets):
    current_itemsets = freq_itemsets
    all_frequent_itemsets = set(current_itemsets)
    cuckoo_hash = defaultdict(int)

    for item in freq_itemsets:
        cuckoo_hash[item] = sum(1 for t in transactions if item.issubset(t))
    
    k = 2
    while current_itemsets:
        candidate_itemsets = generate_candidates(current_itemsets, k)
        candidate_counts = defaultdict(int)

        for transaction in transactions:
            for candidate in candidate_itemsets:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] += 1

        current_itemsets = {item for item, count in candidate_counts.items() if count >= support_threshold}

        for item in current_itemsets:
            cuckoo_hash[item] = candidate_counts[item]

        all_frequent_itemsets.update(current_itemsets)
        k += 1

    return all_frequent_itemsets, cuckoo_hash

def generate_association_rules(all_frequent_itemsets, cuckoo_hash, lift_threshold, confidence_threshold, transactions):
    rules = []
    num_transactions = len(transactions)

    for itemset in all_frequent_itemsets:
        for item in itemset:
            antecedent = frozenset(itemset - {item})
            consequent = frozenset([item])
            
            if antecedent and antecedent in cuckoo_hash and cuckoo_hash[antecedent] > 0:
                confidence = cuckoo_hash[itemset] / cuckoo_hash[antecedent]
                lift = confidence / (cuckoo_hash[consequent] / num_transactions)
                
                if confidence >= confidence_threshold and lift >= lift_threshold:
                    rules.append((antecedent, consequent, confidence, lift))

    print("\nAssociation Rules:")
    for rule in rules:
        print(f"{rule[0]} -> {rule[1]} (Confidence: {rule[2]:.2f}, Lift: {rule[3]:.2f})")

    return rules

def main(file_path):
    df = pd.read_csv(file_path)
    ratings_values = df.drop(columns='user_id').values.flatten()
    skewness = calculate_skewness(ratings_values)
    support_threshold, confidence_threshold, lift_threshold = determine_thresholds(skewness, len(df))

    print(support_threshold)

    bloom_filter = BloomFilter(capacity=100000, error_rate=0.01)
    transactions = convert_to_transactions(df.drop(columns='user_id'))
    
    freq_itemsets = generate_frequent_1_itemsets(transactions, support_threshold, bloom_filter)
    all_frequent_itemsets, cuckoo_hash = mine_frequent_itemsets(transactions, support_threshold, freq_itemsets)

    f = open("results.txt", "w")

    f.write(str(all_frequent_itemsets))
    
    generate_association_rules(all_frequent_itemsets, cuckoo_hash, lift_threshold, confidence_threshold, transactions)

file_path = 'result3.csv'
main(file_path)
