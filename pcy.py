import argparse
from collections import defaultdict
import itertools as it
import pandas as pd
import time
import psutil


# track number of collisions
num_collision = 0
collision_rate = 0
# Track number of placed keys
num_placed_keys = 0

def hash(num1, num2):
    ''' Hash function for the hash table '''
    num1 = int(num1)
    num2 = int(num2)
    return (num1 ^ num2) % 1000

def create_bitmap(hash_table, threshold):
    ''' Convert the hash table into a bitmap '''
    bit_map = [0] * 1000  # Initialize a bitmap
    for key, value in hash_table.items():
        if value >= threshold:
            bit_map[key] = 1
    return bit_map


def create_candidate_item_set(dataset_file):
    ''' Create a dictionary of all candidate item sets from the data set with their corresponding count '''
    global collision_rate
    global num_collision
    global num_placed_keys

    candidate_item_list = defaultdict(int)
    baskets = []
    buckets = {}

    # Read the dataset
    data = pd.read_csv(dataset_file)

    # Convert the data to baskets
    for index, row in data.iterrows():
        basket = [item_id for item_id, rating in row.items() if rating == 1 and item_id != 'user_id']
        baskets.append(basket)

        # Count items
        for item in basket:
            candidate_item_list[item] += 1

       #apply hash to buckets
        pairs = list(it.combinations(basket, 2))
        for pair in pairs:
            index = hash(pair[0], pair[1])

            if index in buckets:
                num_collision += 1
                #collision if the bucket exists
            buckets[index] = buckets.get(index, 0) + 1
            num_placed_keys += 1

    return candidate_item_list, baskets, buckets


def create_frequent_item_set(item_list, min_threshold):
    ''' Return the frequent items from the candidate_item_list that meet the min_support '''

    # items to remove
    to_remove = [key for key, value in item_list.items() if value < min_threshold]

    # Delete items below threshold
    for key in to_remove:
        del item_list[key]

    return item_list.keys()


def count(item_list, baskets):
    ''' Count the number of frequent item sets in the baskets '''
    count = defaultdict(int)

    for basket in baskets:
        for item in item_list:
            if item in basket:
                count[item] += 1

    return count

def get_memory_usage():
    process = psutil.Process()
    memory = process.memory_info().rss / 1024 / 1024
    return memory

def pcy_performance(dataset_file, threshold, output_file = "OriginalPcy.txt"):

    start_time = time.time()

    start_memory = get_memory_usage()
    # candidate itemsets and buckets
    C1, baskets, buckets = create_candidate_item_set(dataset_file)

    # hash table into bitmap
    bitmap = create_bitmap(buckets, threshold)
    # filter the frequent items
    F1_items = create_frequent_item_set(C1, threshold)

    end_time = time.time()
    end_memory = get_memory_usage()

    # Write the output to the file
    with open(output_file, 'w') as f:
        f.write(f'Frequent Item Sets Threshold: {args.threshold}\n')
        f.write(f'Frequent Item Sets: {sorted(F1_items)}\n')
        f.write(f"Original PCY Execution Time: {end_time - start_time:.4f} seconds\n")
        f.write(f"Memory Used: {end_memory - start_memory:.2f} MB\n")
        f.write(f"Collision rate: {(num_collision / num_placed_keys):.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCY Algorithm')
    parser.add_argument('datafile', help='Dataset File (CSV)')
    parser.add_argument('threshold', help='Threshold Value', type=int)

    args = parser.parse_args()
    pcy_performance(args.datafile, args.threshold)

