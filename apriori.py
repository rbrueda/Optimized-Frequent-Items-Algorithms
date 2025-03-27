#problem statement: trying to find which movies are likely to be rated together
#used code from:https://github.com/asaini/Apriori/blob/python3/apriori.py 

#todo: implement with a trie for storing frequent itemsets
#use vertical data representation (apriori-TID)
#apply bloom filters

import pandas as pd
import scipy
from collections import defaultdict
from itertools import  chain, combinations
from optparse import OptionParser
import time

def subsets(arr):
    return chain(*[combinations(arr, i+1) for i, a in enumerate(arr)])

#calculate support for items in the movieSet and returns a subset of the movieSet with  items that meet min support threshold
def returnItemsWithMinSupport(movieSet, ratingsList, minSupport, freqSet):
    print("ratings list")
    print(ratingsList)
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

#todo: figure out what joinSet does

#todo: figutr out what getItemSetRatingList does
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
def apriori(df, minSupport, minConfidence):
    #extracts the set of unique items (movies) and the list of transactions (user ratings or transactions) from the dataset
    movieSet, ratingsList = getMovieSetRatingsList(df)

    freqSet = defaultdict(int) #dictionary to store frequency of itemsets
    largeSet = dict() #stores frequent itemsets at each iteration
    assocRules = dict() #stores the association rules


    #extracts frequent 1-itemsets -> updates freqSet with support counts
    oneCSet = returnItemsWithMinSupport(movieSet, ratingsList, minSupport, freqSet)

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
        currentLSet = returnItemsWithMinSupport(
            currentCSet, ratingsList, minSupport, freqSet)
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
            print("current item:")
            print(item)
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                print("element:")
                print(element)
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item) / getSupport(element)
                    print("current confidence")
                    print(confidence)
                    #filter rules using minconfidence
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)), confidence))
    
    return toRetItems, toRetRules

#todo: find better starting threshold

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

items, rules = apriori(df, 209, confidence_threshold)

end_time = time.time()

elapsed_time = end_time - start_time

with open("apriori-orig-result.txt", "w") as f:
    f.write(f"Items:\n{items}\n---------------------------------------------------------------------\nRules:\n{rules}\n\n")
    f.write(f"final time: {elapsed_time:.4f} seconds")

print("items:")
print(items)
print("...............")
print("rules:")
print(rules)


print(f"final time: {elapsed_time:.4f} seconds")


# apriori(df, support_threshold, lift_threshold)
#got a min threshold to 47 -> make sure to use this

#continue to modify support threshold based on associations

#using lift instead of confidence -> find formula based on skewness


#based on skewness of data -> adjust minSupport
#create a formula for association rules (Lift and Confidence) based on skewness 

#generate confudence threshold to focus on stronger patterns

#low skewness -> fewer rules need filtering 

#apply lift rule evaluation instead of confidence 

#find most optimal size of continegency tables -> fiind most optimized data structure for this

#todo: usingg probabilistic storage techniques -> Bloom filters (track itemset existence)

#iApriori with vertical representation
#incremental learning for updateds rules efficiently

#liift based filter for stroonger wordshtreshold