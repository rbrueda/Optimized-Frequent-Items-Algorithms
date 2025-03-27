#applied dynamic min support - using weighted sum formula
#w1, w2, w3 -> control impact
#higher pruning -> lower 
# adjusted_minSupport = minSupport * (1- (w1*pruning_ratio) - (w2 * transaction_shrinkage) - (w3*|skewness|))
# higher pruning -> lower min support (amount of items taken away are pruned too fast)
# more shrinkage -> lower min support (retains more patterns)
# higher skewness -> lower min suppport (if data is imbalanced it retains more)

#problem statement: trying to find which movies are likely to be rated together
#used code from:https://github.com/asaini/Apriori/blob/python3/apriori.py 

#changes: used lift that provides a better measure of how 2 items are truly independent from each other
#   - ensures only strong associations are kept 

#introduces a dynamic, priority-based pruning mechanism on top of a vertical data representation

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

#implement heap-based pruning -> min heap to prioritize high impact itemsets based on their support -> dynamically remocies low-priority itemsets
#prioritizing items by Lift, Transaction impact
#start initializing empty 
def pruneLowImpactItems(vertical_df, minSupport, total_transactions, transactions_lengths, prev_heap=None):
    #add priority (higher rate to get pruned) to lower lift (weaher association) and lower transaction impact (higher impact means we keep it)
    w1 = 0.5
    w2 = 0.5

    #todo: figure out if we eent need freqSet

    #first iteration -> need to create a new heap
    if prev_heap is None:
        min_heap = [] #min heap for storing (support, item) pairs

        #iterate through each entry in vertical_df (item -> movie, transaction -> user_ids)
        for item, transactions in vertical_df.items():
            #find support
            support = len(transactions)
            if support >= minSupport:
                lift = (support / total_transactions) / ((sum(len(v) for v in vertical_df.values()) / total_transactions) ** 2 )
                #since support is len(transactions)
                impact = support / total_transactions if total_transactions > 0 else 0

                #prioritizing lower lift and lower impact 
                priority = (w1 * lift) + (w2 * (1- impact))
                heapq.heappush(min_heap, (priority, item)) #push items into heap (low-priority -> prune first)
    else:
        min_heap = prev_heap

    #remove low-impact items
    #todo: stop at a certain threshold and report documentation for this implementation
    pruned_df = {}
    
    while min_heap:
        #remove values from min_heap -> we remove values in min heap that ___
        priority, item = heapq.heappop(min_heap)

        #only add values to vertical df where the support >= minSupport
        if len(vertical_df[item]) >= minSupport:
            pruned_df[item] = vertical_df[item]
        
    return pruned_df

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
    #convert to vertical data representation
    vertical_df = getVerticalDataRepresentation(df)

    freqSet = defaultdict(int) #dictionary to store frequency of itemsets
    largeSet = dict() #stores frequent itemsets at each iteration
    assocRules = dict() #stores the association rules

    #find frequent 1-itemset from the vertical data representation
    oneCSet = set()
    for movie, transactions in vertical_df.items():
        if len(transactions) >= minSupport:
            oneCSet.add(frozenset([movie])) #add frequent 1-itemset
            freqSet[frozenset([movie])] = len(transactions)  # update freqSet for single items

    #starting with 1-itemsets as the initial frequent itemsets
    currentLSet = oneCSet
    #total number of occurences before prunning
    original_size = sum(len(v) for v in vertical_df.values()) 

    #store previous frequent itemset size
    prevLSet = oneCSet

    #initial adjusted minsupport is just current one calculated
    adjusted_minSupport = minSupport

    #starts with pairwise combinations
    k = 2
    #iterates umtil no more frequent itemsets can be found
    while currentLSet:
        print(f"currentLSet {k}:")
        print(currentLSet)
        #store frequent itemsets
        largeSet[k-1] = currentLSet
        #generate k-itemset candidates using set union for faster pair generation 
        currentCSet = joinSet(currentLSet, k)

        #Apply pruning using Vertical Data Representation and heap-based pruning
        vertical_df_pruned = pruneLowImpactItems(vertical_df, adjusted_minSupport, len(df), len(df), prev_heap=None)
        current_size = sum(len(v) for v in vertical_df_pruned.values())
        
        # pruning_ratio = current_size / original_size if original_size > 0 else 1
        
        # # Simple minSupport adjustment based on pruning ratio
        # minSupport_floor = 1  # Minimum valid support (to avoid going too low)
        # max_threshold = len(df)  # Cap at the number of users or dataset size

        # # Define how sensitive the adjustment should be based on pruning ratio
        # pruning_factor = 1.5  # You can tweak this value to adjust sensitivity

        # #cache previous minSupport value
        # previous_minSupport = adjusted_minSupport

        # # Adjust minSupport based on pruning ratio
        # if pruning_ratio > 0.5:  # If more than 50% of items are pruned, increase minSupport
        #     adjusted_minSupport = previous_minSupport * pruning_factor
        # else:  # If pruning is not significant, decrease minSupport slightly
        #     adjusted_minSupport = previous_minSupport / pruning_factor

        # # Ensure the adjusted minSupport stays within the practical range
        # adjusted_minSupport = max(minSupport_floor, adjusted_minSupport)  # Don't go below minSupport_floor
        # adjusted_minSupport = min(adjusted_minSupport, max_threshold)  # Don't exceed max threshold (number of users)

        # # Round the result to an integer
        # adjusted_minSupport = round(adjusted_minSupport)

        # # Output the adjusted minSupport
        # print("Adjusted minSupport:", adjusted_minSupport)

        #cache previously frequent itemset
        prevLSet = currentLSet

        #generate k-itemsets and count supports
        currentLSet = set()

        for itemset in currentCSet:
            #calculate support for the vertical dataframe
            transaction_sets = [vertical_df_pruned[item] for item in itemset]
            common_transaction = set.intersection(*transaction_sets)
            support = len(common_transaction)

            # Update freqSet with the calculated support
            freqSet[itemset] = support

            #if the support of the itemset is >= minSupport, add to the currentLSet
            if support >= adjusted_minSupport: 
                currentLSet.add(itemset)
    
        #adjust lift threshold dynamically determined by pruning rate between candidate and freq itemsets
        pruning_effect = len(currentLSet) / len(currentCSet) if len(currentCSet) > 0 else 1
        liftThreshold *= liftThreshold * (1+pruning_effect*  0.3)

        k = k + 1

    #local function that returns the support of an item
    def getSupport(item):
        return freqSet[item]
    
    #store frequnet itemsets into a list with their support values
    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])
    
    #lists the association rules with confidence scores
    #for example {1,2} -> {3}
    #for generating rule filtering
    toRetRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    # Check that the support of element is non-zero to avoid division by zero for confidence
                    support_element = getSupport(element)
                    if support_element > 0:  # Prevent division by zero for confidence calculation
                        confidence = getSupport(item) / getSupport(element)
                        #filter rules using minconfidence
                        if confidence >= minConfidence:
                            toRetRules.append(((tuple(element), tuple(remain)), confidence))
                            lift = confidence / (getSupport(remain) / len(df))
                            if lift >= liftThreshold:
                                toRetRules.append(((tuple(element), tuple(remain)), lift))
    
    return toRetItems, toRetRules

def skewness_correction(skewness, k=2.5, c=1.0):
    return 1 + (np.tanh(k * (abs(skewness) - c)) / 2)

df = pd.read_csv('result2.csv')

# Drop user_id column
ratings = df.drop(columns='user_id')

# Flatten all ratings
ratings_values = ratings.values.flatten()

# Calculate mean, median, standard deviation
mean_item = ratings_values.mean()
median_item = pd.Series(ratings_values).median()
std_item = pd.Series(ratings_values).std()

# Compute skewness
skewness = (3 * (mean_item - median_item)) / std_item

# Compute skewness correction
S_skew = skewness_correction(skewness)

# Compute α dynamically based on dataset size
num_users = len(df)
num_movies = len(ratings.columns)

alpha = 0.1 * np.log10(num_users)  # Scale α based on dataset size

min_support_count = alpha * num_users * S_skew 
min_support_count = int(min_support_count)
#make sure it doesnt exceed the side of the number of users
min_support_count = min(num_users, min_support_count)

# Assign confidence and lift thresholds based on skewness
if skewness > 1.5:
    min_confidence = 0.75  # Stricter confidence for highly skewed data
    min_lift = 1.5         # Require stronger associations
elif 0.5 <= skewness <= 1.5:
    min_confidence = 0.65  # Moderate confidence
    min_lift = 1.2         # Normal lift threshold
else:
    min_confidence = 0.6   # Less strict for low-skew data
    min_lift = 1.0         # Default neutral lift

start_time = time.time()
items, rules = apriori(df, min_support_count, min_confidence, min_lift)
end_time = time.time()

elapsed_time = end_time - start_time

# print("items:")
# print(items)
# print("...............")
# print("rules:")
# print(rules)

with open("apriori-new-res.txt", "w") as f:
    f.write(f"Items:\n{items}\n---------------------------------------------------------------------\nRules:\n{rules}\n\n")
    f.write(f"final time: {elapsed_time:.4f} seconds")

print(f"Skewness: {skewness:.4f}")
print(f"Skewness Correction Factor (S_skew): {S_skew:.4f}")
print(f"Adjusted Min Support Count: {min_support_count}")

print(f"elapsed time {elapsed_time:.4f} seconds")