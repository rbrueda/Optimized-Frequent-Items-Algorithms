#Used original Apriori code from:https://github.com/asaini/Apriori/blob/python3/apriori.py 

import pandas as pd
import scipy
from collections import defaultdict
from itertools import  chain, combinations
from optparse import OptionParser
#for priority structure implementation for vertical data representation
import heapq
import time
import numpy as np

#modify getMovieSetRatingList to creeate vertical data representation
def getVerticalDataRepresentation(df):
    #drop 'user_id' and work only with the ratings for movies
    df = df.drop('user_id', axis=1)
    vertical_df = defaultdict(set) #dict where keys are movies and sets if user IDs (transactions) -- vertical representation

    for index, row in df.iterrows():
        for movie in row.index[row == 1]: #the first row is the list of movies (1...1682 movies)
            vertical_df[movie].add(index) #add the transaction ID (index) to the movie's set

    return vertical_df

#implement heap-based pruning -> min heap to prioritize high impact itemsets based on their support -> dynamically prunes low-priority itemsets
#prioritizing items by importance score and impact score 
import heapq
import numpy as np

def pruneLowImpactItems(vertical_df, minSupport, total_transactions, w1=0.5, w2=0.5):
    min_heap = []
    priority_values = []
    
    #computes median support to define relative importance (only median will values that pass minSupport)
    supports = [len(v) for v in vertical_df.values() if len(v) >= minSupport]
    median_support = np.median(supports) if supports else 1  #avoids division by zero

    for movieID, transactions in list(vertical_df.items()):
        support = len(transactions) #computes importance
        if support >= minSupport:  #only consider items passing the support threshold
            importance_score = support / median_support  #relative to median support
            impact = support / total_transactions if total_transactions > 0 else 0
            priority = (w1 * importance_score) + (w2 * impact)
            priority_values.append(priority)
            heapq.heappush(min_heap, (priority, movieID, transactions))
    
    print(f"Initial heap size: {len(min_heap)}")
    
    if len(priority_values) > 1:
        Q1 = np.percentile(priority_values, 25)
        Q3 = np.percentile(priority_values, 75)
        IQR = Q3 - Q1
        pruning_threshold = Q1 - 1.5 * IQR
        print(f"Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}, Pruning threshold: {pruning_threshold:.4f}")
    else:
        pruning_threshold = float('-inf')  # No pruning if only one item

    pruned_heap = []
    pruned_count = 0
    while min_heap:
        priority, movieID, transactions = heapq.heappop(min_heap)
        if priority >= pruning_threshold:
            heapq.heappush(pruned_heap, (priority, movieID, transactions))
        else:
            pruned_count += 1
    
    print(f"Pruned {pruned_count} items. Remaining heap size: {len(pruned_heap)}")
    vertical_df_pruned = {movieID: transactions for _, movieID, transactions in pruned_heap}
    return vertical_df_pruned


def subsets(arr):
    return chain(*[combinations(arr, i+1) for i, a in enumerate(arr)])

#retuns a n-element itemset
def joinSet(itemSet, length):
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length and i != j]
    )

#integrating VDR and heap pruning into apriori algorithm
#update: use lift instead of confidence
def apriori(df, minSupport, minConfidence, liftThreshold):
    # Convert to vertical data representation
    vertical_df = getVerticalDataRepresentation(df)

    #apply stricter pruning using vertical data representation and heap-based pruning
    vertical_df = pruneLowImpactItems(vertical_df, minSupport, len(df))

    freqSet = defaultdict(int)  #dictionary to store frequency of itemsets
    largeSet = dict()  #stores frequent itemsets at each iteration

    # Find frequent 1-itemset from the vertical data representation
    oneCSet = set()
    for movie, transactions in vertical_df.items():
        oneCSet.add(frozenset([movie]))  # Add frequent 1-itemset
        freqSet[frozenset([movie])] = len(transactions)  # Update freqSet for single items

    # Starting with 1-itemsets as the initial frequent itemsets
    currentLSet = oneCSet

    #gets the original size of the k=1 itemsets
    original_size = sum(len(v) for v in vertical_df.values())

    # Initial adjusted minSupport is just current one calculated
    adjusted_minSupport = minSupport

    #starts with pairwise combinations
    k = 2

    #iterates until no more frequent itemsets can be found
    while currentLSet:
        #stores frequent itemsets
        largeSet[k-1] = currentLSet

        #generates k-itemset candidates using set union for faster pair generation 
        currentCSet = joinSet(currentLSet, k)

        current_size = sum(len(v) for v in vertical_df.values())
        
        pruning_ratio = current_size / original_size if original_size > 0 else 1
        
        # Simple minSupport adjustment based on pruning ratio
        minSupport_floor = 1  # Minimum valid support (to avoid going too low)
        max_threshold = len(df)  # Cap at the number of users or dataset size

        # Define how sensitive the adjustment should be based on pruning ratio
        pruning_factor = 1.5  # You can tweak this value to adjust sensitivity

        #cache previous minSupport value
        previous_minSupport = adjusted_minSupport

        # Adjust minSupport based on pruning ratio
        if pruning_ratio > 0.5:  # If more than 50% of items are pruned, increase minSupport
            adjusted_minSupport = previous_minSupport * pruning_factor
        else:  # If pruning is not significant, decrease minSupport slightly
            adjusted_minSupport = previous_minSupport / pruning_factor

        # Ensure the adjusted minSupport stays within the practical range
        adjusted_minSupport = max(minSupport_floor, adjusted_minSupport)  # Don't go below minSupport_floor
        adjusted_minSupport = min(adjusted_minSupport, max_threshold)  # Don't exceed max threshold (number of users)

        # Round the result to an integer
        adjusted_minSupport = round(adjusted_minSupport)

        # Output the adjusted minSupport
        print("Adjusted minSupport:", adjusted_minSupport)

        #generates k-itemsets and count supports
        currentLSet = set()

        for itemset in currentCSet:
            #calculates support for the vertical dataframe
            transaction_sets = [vertical_df[item] for item in itemset]
            common_transaction = set.intersection(*transaction_sets)
            support = len(common_transaction)

            #update freqSet with the calculated support
            freqSet[itemset] = support

            # If the support of the itemset is >= minSupport, add to the currentLSet
            if support >= adjusted_minSupport: 
                currentLSet.add(itemset)

        print(f"k = {k} {len(currentLSet)}")

        k = k + 1

    #local function that returns the support of an item
    def getSupport(item):
        return freqSet[item]

    #stores frequent itemsets into a list with their support values
    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    #list the association rules with confidence scores
    toRetRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    #check that the support of element is non-zero to avoid division by zero for confidence
                    support_element = getSupport(element)
                    if support_element > 0:  # Prevent division by zero for confidence calculation
                        confidence = getSupport(item) / getSupport(element)
                        #filter rules using minConfidence
                        if confidence >= minConfidence:
                            toRetRules.append(((tuple(element), tuple(remain)), confidence))
                            lift = confidence / (getSupport(remain) / len(df))
                            #lift for stronger rule association
                            if lift >= liftThreshold:
                                toRetRules.append(((tuple(element), tuple(remain)), lift))

    return toRetItems, toRetRules


def skewness_correction(skewness, k=2.5, c=1.0):
    return 1 + (np.tanh(k * (abs(skewness) - c)) / 2)

#reads the binary data of movie rating transactions
df = pd.read_csv('result2.csv')

#drop user_id column
ratings = df.drop(columns='user_id')

#flatten all the ratings
ratings_values = ratings.values.flatten()

#calculate mean, median, standard deviation
mean_item = ratings_values.mean()
median_item = pd.Series(ratings_values).median()
std_item = pd.Series(ratings_values).std()

#compute skewness
skewness = (3 * (mean_item - median_item)) / std_item

#compute skewness correction
S_skew = skewness_correction(skewness)

#compute α dynamically based on dataset size
num_users = len(df)
num_movies = len(ratings.columns)

alpha = 0.1 * np.log10(num_users)  #scale α based on dataset size

min_support_count = alpha * num_users * S_skew 
min_support_count = int(min_support_count)
#make sure it doesnt exceed the side of the number of users
min_support_count = min(num_users, min_support_count)

#assign confidence and lift thresholds based on skewness
if skewness > 1.5:
    min_confidence = 0.75  # Stricter confidence for highly skewed data
    min_lift = 1.5         # Require stronger associations
elif 0.5 <= skewness <= 1.5:
    min_confidence = 0.65  # Moderate confidence
    min_lift = 1.2         # Normal lift threshold
else:
    min_confidence = 0.6   # Less strict for low-skew data
    min_lift = 1.0         # Default neutral lift


print(f"Skewness: {skewness:.4f}")
print(f"Skewness Correction Factor (S_skew): {S_skew:.4f}")
print(f"Adjusted Min Support Count: {min_support_count}")


start_time = time.time()
items, rules = apriori(df, 209, min_confidence, min_lift)
end_time = time.time()

elapsed_time = end_time - start_time

with open("apriori-new-res.txt", "w") as f:
    f.write(f"Items:\n{items}\n---------------------------------------------------------------------\nRules:\n{rules}\n\n")
    f.write(f"final time: {elapsed_time:.4f} seconds")

print(f"elapsed time {elapsed_time:.4f} seconds")