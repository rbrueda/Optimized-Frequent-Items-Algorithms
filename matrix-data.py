#code from documentation: https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html

import pandas as pd

#create a pandas dataframe from original data
data = pd.read_csv(
    "u.data", sep="\t", usecols=[0, 1, 2], names=["user_id", "item_id", "rating"]
)

#convert to sparse matrix (Pivot Table)
sparse_matrix = data.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc=lambda x: 1).fillna(0)

#convert to integer type (1 if watched/rated, 0 if not)
sparse_matrix = sparse_matrix.astype(int)

#reset index to include 'user_id' as a column instead of an index
sparse_matrix.reset_index(inplace=True)

sparse_matrix.to_csv('result2.csv', index=False)
