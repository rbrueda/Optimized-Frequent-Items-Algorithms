import argparse
from collections import defaultdict
import itertools as it
import pandas as pd
import time
import psutil


maxN = 10000  #intial hash table size
hTables = 2 # 2 hash tables for cuckoo hashing


# Initialize hash tables
hashtable = [[float('inf')] * maxN for _ in range(hTables)]
pos = [0] * hTables  # To store possible positions for a key

# track number of collisions
num_collision = 0
collision_rate = 0
# Track number of placed keys
num_placed_keys = 0


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
       return (key // maxN) % maxN
   else:
       raise ValueError(f"unknown hash table number: {table_num}")


def placeT(key, table_num, count, n):
   """Place a key in one of its possible positions in the hash table"""
   global num_placed_keys
   global num_collision
   global collision_rate


   if count == n:
       print(f"{key} unpositioned")
       return


   for i in range(hTables):
       pos[i] = hash(i, key)
       # Ensure pos[i] is a valid index
       if pos[i] is None:
           raise ValueError(f"Invalid hash position for key {key} in hash table {i}")


       if hashtable[i][pos[i]] == key:
           #print(f"Key {key} already placed at table {i}, position {pos[i]}")
           return  # Key has been positioned already

   #check if a key was positioned at place of new key to be positioned
   if hashtable[table_num][pos[table_num]] != float('inf'):
       displaced_key = hashtable[table_num][pos[table_num]]
       #print(f"Displaced key: {displaced_key} from table {table_num} at position {pos[table_num]}")
       hashtable[table_num][pos[table_num]] = key
       num_collision += 1
       #to recursively place the displaced key
       placeT(displaced_key, (table_num + 1) % hTables, count + 1, n)
   else:
       #otherwise place key in the position
       #print(f"Placing key {key} at table {table_num}, position {pos[table_num]}")
       hashtable[table_num][pos[table_num]] = key
       num_placed_keys += 1


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


   # Convert the data to baskets (rows with 1's indicate the items that were watched/rated by the user)
   for index, row in data.iterrows():
       basket = [item_id for item_id, rating in row.items() if rating == 1 and item_id != 'user_id']
       baskets.append(basket)

       # Count items
       for item in basket:
           candidate_item_list[item] += 1
       pairs = list(it.combinations(basket, 2))


       #using cuckoo hashing to add pairs to the hash tables
       for pair in pairs:

           placeT(pair[0], 0, 0, hTables) #place first item in hash table 0
           placeT(pair[1], 1, 0, hTables) #place second item in hash table 1

           index1 = hash(0, pair[0])
           index2 = hash(1, pair[1])

           buckets[index1] = 1 if index1 not in buckets else buckets[index1] + 1
           buckets[index2] = 1 if index2 not in buckets else buckets[index2] + 1


   return candidate_item_list, baskets, buckets
def create_frequent_item_set(item_list, min_threshold):
   ''' Return the frequent items from the candidate_item_list that meet the min_support '''
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


def pcy_performance(dataset_file, threshold, output_file="PcyWithCuckoo.txt"):
   global num_collision

   # Start time measurement
   start_time = time.time()
   start_memory = get_memory_usage()

   # candidate item sets and buckets
   C1, baskets, buckets = create_candidate_item_set(dataset_file)


   # convert hash table into bitmap
   bitmap = create_bitmap(buckets, threshold)
   # filter the frequent items
   F1_items = create_frequent_item_set(C1, threshold)

   end_time = time.time()
   end_memory = get_memory_usage()


   # Write output to the file
   with open(output_file, 'w') as f:
       f.write(f'Frequent Item Sets Threshold: {args.threshold}\n')
       f.write(f'Frequent Item Sets: {sorted(F1_items)}\n')
       f.write(f" PCY with Cuckoo Hashing Execution Time: {end_time - start_time:.4f} seconds\n")
       f.write(f"Memory Used: {end_memory - start_memory:.2f} MB\n")
       f.write(f"Collision rate: {(num_collision / num_placed_keys):.4f}\n")

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='PCY Algorithm with Cuckoo Hashing')
   parser.add_argument('datafile', help='Dataset File (CSV)')
   parser.add_argument('threshold', help='Threshold Value', type=int)

   args = parser.parse_args()
   pcy_performance(args.datafile, args.threshold)



