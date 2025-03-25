#problem statement: trying to find which movies are likely to be rated together
#used code from:https://github.com/asaini/Apriori/blob/python3/apriori.py 

#changes: used lift that provides a better measure of how 2 items are truly independent from each other
#   - ensures only stroong associations are kept



import pandas as pd
import scipy
from collections import defaultdict
from itertools import  chain, combinations
from optparse import OptionParser

def subsets(arr):
    return chain(*[combinations(arr, i+1) for i, a in enumerate(arr)])

#calculate support for items in the movieSet and returns a subset of the movieSet with  items that meet min support threshold
def returnItemsWithMinSupport(movieSet, ratingsList, minSupport, freqSet):
    minimizedMovieSet = set() #generate a new set
    localSet = defaultdict(int)

    for movie in movieSet: 
        for ratings in ratingsList:
            if movie.issubset(ratings):
                freqSet[movie] += 1
                localSet[movie] += 1

    #get the size of pruned items
    pruned_size = sum(localSet.values())
    original_size = len(ratingsList)
    #compute the pruning ratio based on original list of reviews (aka transactions) and # of pruned values
    pruning_ratio = pruned_size / original_size if original_size > 0 else 1

    #dynamic minSupport  adjustment
    adjusted_minSupport = minSupport * (1 - pruning_ratio * 0.5)

    for movie, count in localSet.items():
        support = float(count)

        if support >= minSupport:
            minimizedMovieSet.add(movie) #our new set
        
    return minimizedMovieSet, pruning_ratio

#datapreprocessing -> drop first column
#take our df -> create  2 lists -> 1 with list of rating combinations, df.iterrows() -> perform lambda on this. 2nd with list of movies (iterate through all columns instead of first one)
#issue with this -> extra memory
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
def apriori(df, minSupport, minConfidence, liftThreshold):
    #extracts the set of unique items (movies) and the list of transactions (user ratings or transactions) from the dataset
    movieSet, ratingsList = getMovieSetRatingsList(df)

    freqSet = defaultdict(int) #dictionary to store frequency of itemsets
    largeSet = dict() #stores frequent itemsets at each iteration
    assocRules = dict() #stores the association rules


    #extracts frequent 1-itemsets -> updates freqSet with support counts
    oneCSet, pruning_ratio = returnItemsWithMinSupport(movieSet, ratingsList, minSupport, freqSet)

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

        #prune itemsets using minSupport
        currentLSet, pruning_ratio = returnItemsWithMinSupport(
            currentCSet, ratingsList, minSupport, freqSet)
        
        #stronger pruning using lift association rule mining
        liftThreshold = liftThreshold * (1+pruning_ratio *  0.3)

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
                    confidence = getSupport(item) / getSupport(element)
                    #filter rules using minconfidence
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)), confidence))
                        lift = confidence / (getSupport(remain) / len(ratingsList))
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