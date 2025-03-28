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
import heapq
import numpy as np

def pruneLowImpactItems(vertical_df, minSupport, total_transactions, w1=0.5, w2=0.5, pruning_threshold=None, prev_heap=None):
    """
    Prune low-impact items based on lift and transaction impact dynamically with variable pruning.
    The number of pruned items adjusts depending on how different the priority values are.

    :param vertical_df: Vertical data representation (dict of items -> set of transactions)
    :param minSupport: Minimum support threshold
    :param total_transactions: Total number of transactions in the dataset
    :param w1: Weight for lift prioritization
    :param w2: Weight for transaction impact prioritization
    :param pruning_threshold: Pruning cutoff threshold based on priority values (None to calculate dynamically)
    :return: Pruned vertical data representation
    """
    # Create a min-heap for prioritizing low-lift and low-impact itemsets
    min_heap = []
    priority_values = []

    for item, transactions in vertical_df.items():
        support = len(transactions)
        if support >= minSupport:  # Only consider items with support greater than minSupport
            lift = (support / total_transactions) / (sum(len(v) for v in vertical_df.values()) / total_transactions) ** 2
            impact = support / total_transactions if total_transactions > 0 else 0
            # Priority for pruning: low lift and low impact are prioritized
            priority = (w1 * lift) + (w2 * (1 - impact))
            priority_values.append(priority)
            heapq.heappush(min_heap, (priority, item))

    # If no pruning threshold is provided, calculate it dynamically based on priority value differences
    if pruning_threshold is None:
        # Calculate the difference between consecutive priority values
        priority_diffs = np.diff(sorted(priority_values))
        
        # Determine a threshold based on the average jump size
        avg_jump = np.mean(priority_diffs) if len(priority_diffs) > 0 else 0
        pruning_threshold = np.percentile(priority_diffs, 75)  # Dynamic pruning threshold based on high jumps

    print(f"Using dynamic pruning threshold: {pruning_threshold}")

    # Prune low-priority items with early termination based on priority and pruning threshold
    pruned_df = {}
    while min_heap:
        # Get the item with the lowest priority
        priority, item = heapq.heappop(min_heap)

        # Stop pruning if the priority value is above the pruning threshold
        if priority >= pruning_threshold:
            pruned_df[item] = vertical_df[item]
        else:
            # Stop if we hit an item with priority below threshold
            break

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
    # Convert to vertical data representation
    vertical_df = getVerticalDataRepresentation(df)

    # Apply stricter pruning using vertical data representation and heap-based pruning
    vertical_df = pruneLowImpactItems(vertical_df, minSupport, len(df), len(df), prev_heap=None)

    freqSet = defaultdict(int)  # Dictionary to store frequency of itemsets
    largeSet = dict()  # Stores frequent itemsets at each iteration

    # Find frequent 1-itemset from the vertical data representation
    oneCSet = set()
    for movie, transactions in vertical_df.items():
        oneCSet.add(frozenset([movie]))  # Add frequent 1-itemset
        freqSet[frozenset([movie])] = len(transactions)  # Update freqSet for single items

    # Starting with 1-itemsets as the initial frequent itemsets
    currentLSet = oneCSet
    prevLSet = oneCSet

    # Initial adjusted minSupport is just current one calculated
    adjusted_minSupport = minSupport

    # Starts with pairwise combinations
    k = 2

    # Iterates until no more frequent itemsets can be found
    while currentLSet:
        print(f"currentLSet {k}:")
        print(currentLSet)
        # Store frequent itemsets
        largeSet[k-1] = currentLSet

        # Generate k-itemset candidates using set union for faster pair generation 
        currentCSet = joinSet(currentLSet, k)

        current_size = sum(len(v) for v in vertical_df.values())
        print(f"Pruned vertical data representation size: {current_size}")

        # Generate k-itemsets and count supports
        currentLSet = set()

        for itemset in currentCSet:
            # Calculate support for the vertical dataframe
            transaction_sets = [vertical_df[item] for item in itemset]
            common_transaction = set.intersection(*transaction_sets)
            support = len(common_transaction)

            # Update freqSet with the calculated support
            freqSet[itemset] = support

            # If the support of the itemset is >= minSupport, add to the currentLSet
            if support >= adjusted_minSupport: 
                currentLSet.add(itemset)
        
        # Adjust lift threshold dynamically determined by pruning rate between candidate and freq itemsets
        pruning_effect = len(currentLSet) / len(currentCSet) if len(currentCSet) > 0 else 1
        liftThreshold *= liftThreshold * (1 + pruning_effect * 0.3)

        k = k + 1

    # Local function that returns the support of an item
    def getSupport(item):
        return freqSet[item]

    # Store frequent itemsets into a list with their support values
    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    # List the association rules with confidence scores
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
                        # Filter rules using minConfidence
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

with open("apriori-new-res.txt", "w") as f:
    f.write(f"Items:\n{items}\n---------------------------------------------------------------------\nRules:\n{rules}\n\n")
    f.write(f"final time: {elapsed_time:.4f} seconds")

print(f"Skewness: {skewness:.4f}")
print(f"Skewness Correction Factor (S_skew): {S_skew:.4f}")
print(f"Adjusted Min Support Count: {min_support_count}")

print(f"elapsed time {elapsed_time:.4f} seconds")