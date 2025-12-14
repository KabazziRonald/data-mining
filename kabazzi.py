import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Part A: Data Preparation
# Creating the Dataset
data = {
    'Transaction_ID': [1,2,3,4,5,6,7,8,9,10],
    'Items': [
        ["Bread","Milk","Eggs"],
        ["Bread","Butter"],
        ["Milk","Diapers","Beer"],
        ["Bread","Milk","Butter"],
        ["Milk","Diapers","Bread"],
        ["Beer","Diapers"],
        ["Bread","Milk","Eggs","Butter"],
        ["Eggs","Milk"],
        ["Bread","Diapers","Beer"],
        ["Milk","Butter"]
    ]
}

# Creating DataFrame
df = pd.DataFrame(data)
print("\n Original Dataset")
print(df)

# Converting to list-of-lists for Apriori
transactions = df['Items'].tolist()

# One-hot encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print("\n One-Hot Encoded Transactions")
print(df_encoded)

# Part B: Apriori Algorithm

# Minimum support = 0.2
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

print("\nFrequent Itemsets (Support ≥ 0.2)")
print(frequent_itemsets)

# Generating rules with confidence ≥ 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Selecting only useful columns
rules = rules[['antecedents','consequents','support','confidence','lift']]

print("\nAssociation Rules (Conf ≥ 0.5)")
print(rules)
