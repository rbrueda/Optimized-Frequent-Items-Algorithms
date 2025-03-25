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
#for priority strcuture impleemtnation for vertical data representation
import heapq

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
def pruneLowImpactItems(vertical_df, minSupport):
    min_heap = [] #min heap for storing (support, item) pairs

    #iterate through each entry in vertical_df (item -> movie, transaction -> user_ids)
    for item, transactions in vertical_df.items():
        support = len(transactions) #number of people rated the movie
        if support >= minSupport:
            heapq.heappush(min_heap, (support, item)) #push items into heap

    #remove low-impact items
    pruned_df = {}
    
    while min_heap:
        #remove values from min_heap -> we remove values in min heap that ___
        support, item = heapq.heappop(min_heap)

        #only add values to vertical df where the support >= minSupport
        if support >= minSupport:
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

    #find frequent 1-tiemset from the vertical data representation
    oneCSet = set()
    for movie, transactions in vertical_df.items():
        if len(transactions) >= minSupport:
            oneCSet.add(frozenset([movie])) #add frequent 1-itemset
            freqSet[frozenset([movie])] = len(transactions)  # update freqSet for single items


    #starting with 1-itemsets as the initial frequent itemsets
    currentLSet = oneCSet

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

        #apply pruning using VDR and heap-based pruning
        vertical_df_pruned = pruneLowImpactItems(vertical_df, minSupport)

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
            if support >= minSupport: 
                currentLSet.add(itemset)

        
        #stronger pruning using lift association rule mining
        pruning_ratio = len(currentLSet) / len(currentCSet) if len(currentCSet) > 0 else 1
        liftThreshold *= liftThreshold * (1+pruning_ratio *  0.3)

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

df = pd.read_csv('result3.csv')

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

items, rules = apriori(df, 3, confidence_threshold, lift_threshold)

print("items:")
print(items)
print("...............")
print("rules:")
print(rules)