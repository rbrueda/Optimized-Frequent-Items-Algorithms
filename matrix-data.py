import scipy #for mathematical computations 
import os
import urllib.request
import tiledb
import pandas as pd

from tiledb.ml.readers.pytorch import PyTorchTileDBDataLoader
from tiledb.ml.readers.types import ArrayParams


data = pd.read_csv(
    "u.data", sep="\t", usecols=[0, 1, 2], names=["user_id", "item_id", "rating"]
)
# data_one_hot = pd.get_dummies(data, columns=["user_id", "item_id"])
# user_movie = data_one_hot[data_one_hot.columns.difference(["rating"])]
# ratings = data["rating"]


# Convert to sparse matrix (Pivot Table)
sparse_matrix = data.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc=lambda x: 1).fillna(0)

# Convert to integer type (1 if watched/rated, 0 if not)
sparse_matrix = sparse_matrix.astype(int)

# Reset index to include 'user_id' as a column instead of an index
sparse_matrix.reset_index(inplace=True)

sparse_matrix.to_csv('result2.csv', index=False)
