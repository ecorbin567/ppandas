import pandas as pd
import os
import sys

module_path = os.path.abspath(os.path.join(".."))
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from ppandas import PDataFrame


df1 = pd.read_csv("data/numerical-1.csv")
df2 = pd.read_csv("data/numerical-2.csv")

pd1 = PDataFrame(["Gender", "Age"], df1)
pd2 = PDataFrame(["Gender", "Age"], df2)

print("pd1")
print(pd1.bayes_net.get_cpds(node="Age"))
print(pd1.bayes_net.get_cpds(node="Gender"))
print(pd1.bayes_net.get_cpds(node="Gun Control"))


print("pd2")
print(pd2.bayes_net.get_cpds(node="Age"))
print(pd2.bayes_net.get_cpds(node="Gender"))
print(pd2.bayes_net.get_cpds(node="Gun Control"))


pd_join = pd1.pjoin(pd2, mismatches={"Age": "numerical"})


print("joined")
print(pd_join.bayes_net.get_cpds(node="Age"))
print(pd_join.bayes_net.get_cpds(node="Gender"))
print(pd_join.bayes_net.get_cpds(node="Gun Control"))


# print(pd1.bayes_net.get_cpds(node = "Gun Control"))
# pd_join.visualise(show_tables = True)
queryResult = pd_join.query(["Gun Control"], {"Gender": "female", "Age": "[40,60)"})
print("conditional query")
print(queryResult)

# queryResult = pd_join.query(['Gun Control'])
# print('overall query')
# print(queryResult)
# pd1.visualise(show_tables = True)
