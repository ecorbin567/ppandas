# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:00:14 2019

@author: Alex
"""
# from Latest import p_frame
import pandas as pd
import os
import sys

module_path = os.path.abspath(os.path.join(".."))
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from ppandas import PDataFrame

# import warnings
# warnings.filterwarnings("ignore")


df1 = pd.read_csv("data/trueSample.csv")
df2 = pd.read_csv("data/biasSample.csv")

pd1 = PDataFrame(["Gender", "Age"], df1)
pd2 = PDataFrame(["Gender", "Age"], df2)

pd_join = pd1.pjoin(pd2)

pd1.visualise()

queryResult = pd1.query(["Gun Control"], {"Gender": "Female", "Age": "17 to 48"})
queryResult = pd1.query(["Gun Control"])
# queryResult = pd1.query('Gun Control',show=True)


# pd1.query('Gun Control',show=True)


# d1 = {'Gender':1}
# print(pd1.bayes_net.get_cpds('Gun Control').variable)
# pd1.getAssignments('Age')

# print(pd1.query('Gun Control',d1))
# print(pd1.query('Gun Control'))
# print(pd1.query('Age'))
