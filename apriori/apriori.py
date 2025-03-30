#Used original Apriori code from:https://github.com/asaini/Apriori/blob/python3/apriori.py 

import pandas as pd
import scipy
from collections import defaultdict
from itertools import  chain, combinations
from optparse import OptionParser
import time
import numpy as np

def subsets(arr):
    return chain(*[combinations(arr, i+1) for i, a in enumerate(arr)])

#calculate support for items in the movieSet and returns a subset of the movieSet with  items that meet min support threshold
def returnItemsWithMinSupport(movieSet, ratingsList, minSupport, freqSet):
    minimizedMovieSet = set() #generate a new set
    localSet = defaultdict(int)

    for movie in movieSet: 
        for ratings in ratingsList:
            #if the movie value is 1
            if movie.issubset(ratings):
                freqSet[movie] += 1
                localSet[movie] += 1


    for movie, count in localSet.items():
        support = float(count)

        if support >= minSupport:
            minimizedMovieSet.add(movie) #our new set
        
    return minimizedMovieSet

def getMovieSetRatingsList(df):
    #drop dirst column -> onky keep movie columns
    df =  df.drop('user_id', axis=1)

    ratingsList = list()
    movieSet = set()

    #get the column values in dataframe 
    for index, row in df.iterrows():
        ratings = frozenset(row.index[row==1]) #get movies with a rating (1)
        ratingsList.append(ratings)

        #generate 1-itemSets
        for movie in ratings:
            movieSet.add(frozenset([movie]))

    return movieSet, ratingsList

#retuns a n-element itemset
def joinSet(itemSet, length):
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


#update: use lift instead of confidence
def apriori(df, minSupport, minConfidence):
    #extracts the set of unique items (movies) and the list of transactions (user ratings or transactions) from the dataset
    movieSet, ratingsList = getMovieSetRatingsList(df)

    freqSet = defaultdict(int) #dictionary to store frequency of itemsets
    largeSet = dict() #stores frequent itemsets at each iteration
    assocRules = dict() #stores the association rules

    #extracts frequent 1-itemsets -> updates freqSet with support counts
    oneCSet = returnItemsWithMinSupport(movieSet, ratingsList, minSupport, freqSet)

    print(f"length of k = 1: {len(oneCSet)}")

    #starting with 1-itemsets as the initial frequent itemsets
    currentLSet = oneCSet

    #starts with pairwise combinations
    k = 2
    #iterates umtil no more frequent itemsets can be found
    while currentLSet:
        #store frequent itemsets
        largeSet[k-1] = currentLSet
        #generate k-itemset candidates using set union for faster pair generation 
        currentCSet = joinSet(currentLSet, k)

        #prune itemsets using minSupport
        currentLSet = returnItemsWithMinSupport(
            currentCSet, ratingsList, minSupport, freqSet)
        
        print(f"length of k = {k}: {len(currentLSet)}")

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
    toRetRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item) / getSupport(element)
                    #filter rules using minconfidence
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)), confidence))
    
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
items, rules = apriori(df, min_support_count, min_confidence)
end_time = time.time()

elapsed_time = end_time - start_time

with open("apriori-orig-result.txt", "w") as f:
    f.write(f"Items:\n{items}\n---------------------------------------------------------------------\nRules:\n{rules}\n\n")
    f.write(f"final time: {elapsed_time:.4f} seconds")

print(f"final time: {elapsed_time:.4f} seconds")
