# Optimized-Frequent-Items-Algorithms
Project for COMP-4250

This repository contains the optimized algorithms for both Apriori and PCY. Before you run, make sure all libraries are installed using **pip**.

## Apriori
- Running the results using **results2.csv**
- Files:
  - apriori.py: original apriori code
  - apriori-dynamic.py: added adaptive threshold per kth iteration (not optimal)
  - apriori-optimized.py: optimized apriori with dynamic threshold, heap-based pruning, and Vertical Data Representation (VDR) 
- For optimized Apriori, run the following commands:
```bash
cd apriori
```
```bash
python3 apriori-optimized.py
```

## PCY
- Running the result using dataset.csv
- - Files:
  - pcy.py: original PCY code
  - .py: PCY with cuckoo hashing (not optimized)
  - pcyCuckoo-resizing.py: PCY with dynamic resizing
- For optimized PCY, run the following commands:
```bash
cd PCY
```
- make sure to replace **100** with the support threshold you want to use
```bash
python3 pcyCuckoo-resizing.py dataset.csv 100
```


