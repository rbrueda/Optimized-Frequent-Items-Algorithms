import argparse
from collections import defaultdict
import itertools as it
import pandas as pd
import time
import psutil

#intial hash table size
maxN = 1000
hTables = 2 # 2 hash tables for cuckoo hashing

# Initialize hash tables
hashtable = [[float('inf')] * maxN for _ in range(hTables)]
pos = [0] * hTables  # To store possible positions for a key

# track number of collisions
num_collision = 0
collision_rate = 0
# Track number of placed keys
num_placed_keys = 0

#set up a threshold factor for dynamic resizing
threshold_factor = 0.5

def hash(table_num, key):
    ''' Hash function for the hash tables '''
    # Ensure that key is a valid integer
    if key is None:
        raise ValueError("Key can't be None.")

    key = int(key)

    if table_num == 0:
        #first hash function
        return (key) % maxN
    elif table_num == 1:
        #second hash function
        return ((key//maxN)%maxN)
    else:
        raise ValueError(f"hash error")

def resize():
    global maxN
    global hashtable
    global num_placed_keys

    original_hashtable = hashtable
    oMaxN = maxN
    maxN *= 2
    num_placed_keys = 0

    # new hash table
    hashtable = [[float('inf')] * maxN for _ in range(hTables)]

    # Rehash
    for table_num in range(hTables):
        for key in original_hashtable[table_num]:
            if key != float('inf'):
                placeT(key, table_num, 0, oMaxN)


def placeT(key, table_num, count, n):
    """Place a key in one of its possible positions in the hash table"""
    global num_placed_keys
    global num_collision
    global collision_rate

    if count == n:
        print(f"{key} unpositioned")
        resize()
        return

    for i in range(hTables):
        pos[i] = hash(i, key)

        if hashtable[i][pos[i]] == key:
            return

    #check if a key was positioned at place of new key to be positioned
    if hashtable[table_num][pos[table_num]] != float('inf'):
        displaced_key = hashtable[table_num][pos[table_num]]
        hashtable[table_num][pos[table_num]] = key
        num_collision += 1

        #to recursively place the displaced key
        placeT(displaced_key, (table_num + 1) % hTables, count + 1, n)
    else:
        #otherwise place key in the position
        hashtable[table_num][pos[table_num]] = key
        num_placed_keys += 1

    if num_placed_keys > threshold_factor * maxN:
        resize()

    if num_placed_keys > 0:
        collision_rate = num_collision / num_placed_keys
    else:
        collision_rate = -1

def create_bitmap(hash_table, threshold):
    ''' Convert the hash table into a bitmap '''
    max_keys = max(hash_table.keys())
    bit_map = [0] * (max_keys + 1)
    for key, value in hash_table.items():
        if value >= threshold:
            bit_map[key] = 1
    return bit_map

def create_candidate_item_set(dataset_file):
    ''' Create a dictionary of all candidate item sets from the data set with their corresponding count '''

    candidate_item_list = defaultdict(int)
    baskets = []
    buckets = {}

    # Read the dataset
    data = pd.read_csv(dataset_file)

    # Convert data to baskets
    for index, row in data.iterrows():
        basket = [item_id for item_id, rating in row.items() if rating == 1 and item_id != 'user_id']
        baskets.append(basket)

        # count items
        for item in basket:
            candidate_item_list[item] += 1

        # Create pairs of unique items in each basket and apply hash function to buckets
        pairs = list(it.combinations(basket, 2))

        #using cuckoo hashing
        for pair in pairs:
            placeT(pair[0], 0, 0, hTables) #place first item in hash table 0
            placeT(pair[1], 1, 0, hTables) #place second item in hash table 1

            #update buckets
            index1 = hash(0, pair[0])
            index2 = hash(1, pair[1])

            buckets[index1] = 1 if index1 not in buckets else buckets[index1] + 1
            buckets[index2] = 1 if index2 not in buckets else buckets[index2] + 1

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

# Function to track memory usage
def get_memory_usage():
    process = psutil.Process()
    memory = process.memory_info().rss / 1024 / 1024
    return memory

def pcy_performance(dataset_file, threshold, output_file="PcyCuckoo&Resizing.txt"):
    global num_collision

    start_time = time.time()
    start_memory = get_memory_usage()


    # candidate item sets and buckets
    C1, baskets, buckets = create_candidate_item_set(dataset_file)

    # convert hash table into bitmap
    bitmap = create_bitmap(buckets, threshold)
    #filter the frequent items
    F1_items = create_frequent_item_set(C1, threshold)
    end_time = time.time()
    end_memory = get_memory_usage()

    # Write output to the file
    with open(output_file, 'w') as f:
        f.write(f'Frequent Item Sets Threshold: {args.threshold}\n')
        f.write(f'Frequent Item Sets: {sorted(F1_items)}\n')
        f.write(f"PCY with Cuckoo Hashing & Dynamic Resizing Execution Time: {end_time - start_time:.4f} seconds\n")
        f.write(f"Memory Used: {end_memory - start_memory:.2f} MB\n")
        f.write(f"Collision rate: {(num_collision / num_placed_keys):.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCY Algorithm with Cuckoo Hashing')
    parser.add_argument('datafile', help='Dataset File (CSV)')
    parser.add_argument('threshold', help='Threshold Value', type=int)

    args = parser.parse_args()
    pcy_performance(args.datafile, args.threshold)

