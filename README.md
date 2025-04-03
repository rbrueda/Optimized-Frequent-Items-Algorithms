# Optimized-Frequent-Items-Algorithms
Project for COMP-4250

This repository contains the optimized algorithms for both Apriori and PCY. Before you run, make sure all libraries are installed using **pip**.

- Both frequent algorithms ran on [MovieLens 100K Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset) from Kaggle. The data was transformed in a user-item interaction matrix in a binary format (1 if a user rated a movie, 0 otherwise). The transformed file is written in **result2.csv**.

## Apriori
- Files:
  - **apriori.py**: original Apriori code
  - **apriori-dynamic.py**: added adaptive threshold per kth iteration (not optimal)
  - **apriori-optimized.py**: optimized Apriori with dynamic threshold, heap-based pruning, and Vertical Data Representation (VDR) 
- For optimized Apriori, run the following commands (note: change python file name if you want to run the other Apriori variants):
```bash
cd apriori
```
```bash
python3 apriori-optimized.py
```

## PCY
- Files:
  - **pcy.py**: original PCY code
  - **pcyC.py**: PCY with cuckoo hashing (not optimized)
  - **pcyCuckoo-resizing.py**: PCY with dynamic resizing
- For optimized PCY, run the following commands (note: change python file name if you want to run the other PCY variants):
```bash
cd PCY
```
- make sure to replace **100** with the support threshold you want to use
```bash
python3 pcyCuckoo-resizing.py ../result2.csv 100
```


