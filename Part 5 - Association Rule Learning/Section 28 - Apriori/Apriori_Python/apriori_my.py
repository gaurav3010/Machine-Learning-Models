# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
items = []
for i in range(0, 7501):
    items = []
    for j in range(0, 20):
        items.append(str(dataset[j][i]))
    transactions.append(items)
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_left = 2)

# Visualising the results
results = list(rules)
