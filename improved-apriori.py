import pandas as pd
import scipy

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
elif skewness <= 1.5 and skewness >= 0.5: 
    min_support = 0.05
else:
    min_support = 0.02

support_threshold = int(min_support * len(df))
print(f"support threshold: {support_threshold}")

#based on skewness of data -> adjust minSupport
#create a formula  for  iniital support 

#apply lift rule evaluation instead of confidence 

#find most optimal size of continegency tables -> fiind most optimized data structure for this

#todo: usingg probabilistic storage techniques -> Bloom filters (track itemset existence)

#iApriori with vertical representation
#incremental learning for updateds rules efficiently

#liift based filter for stroonger words