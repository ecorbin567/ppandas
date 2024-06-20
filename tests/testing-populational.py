import pandas as pd
import os
import sys

module_path = os.path.abspath(os.path.join(".."))
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from ppandas import PDataFrame

df1 = pd.read_csv("data/populational1.csv")
df1 = df1.drop(columns=["Gender"])
pd1 = PDataFrame.from_populational_data(["Age"], df1, 600)
pd1.visualise(show_tables=True)
