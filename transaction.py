# pip install mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
# pip install pyfpgrowth
import pyfpgrowth
import pandas as pd

# 2.1 建立交易紀錄表格
transactions = [
    ['A', 'B'],
    ['B', 'C', 'D'],
    ['A', 'C', 'D', 'E'],
    ['A', 'D', 'E'],
    ['A', 'B', 'C'],
    ['A', 'B', 'C', 'D'],
    ['B', 'C'],
    ['A', 'B', 'C'],
    ['A', 'B', 'D'],
    ['B', 'C', 'E']
]

# 2.2 使用 mlxtend.frequent_patterns 的 apriori 和 association_rules 函數
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(frequent_itemsets)
print(rules)

# 2.3 使用 pyfpgrowth 的 find_frequent_patterns 和 generate_association_rules 函數
patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
print(patterns)
print(rules)