import pandas as pd
from ppandas.p_frame import PDataFrame
import pytest
import traceback
"""
Tests (re)written by me (Elise) for ppandas scenarios relevant to piccard :)
"""

def test_pjoin_categorical_mismatch():
    df1 = pd.DataFrame({
        'cat': ['df1_Q1', 'df1_Q2', 'df1_Q3', 'df1_Q1', 'df1_Q2', 'df1_Q3'],
        'val': [1, 2, 3, 4, 5, 6]
    })
    df2 = pd.DataFrame({
        'cat': ['df2_Q1', 'df2_Q2', 'df2_Q3', 'df2_Q1', 'df2_Q2', 'df2_Q3'],
        'val': [10, 20, 30, 40, 50, 60]
    })

    pdf1 = PDataFrame(independent_vars=['cat'], data=df1)
    pdf2 = PDataFrame(independent_vars=['cat'], data=df2)

    joined = pdf1.pjoin(pdf2, mismatches={'cat': 'categorical'})

def test_categorical():
    df1 = pd.read_csv("https://raw.githubusercontent.com/ecorbin567/ppandas/refs/heads/master/tests/data/numerical-1.csv")
    df2 = pd.read_csv("https://raw.githubusercontent.com/ecorbin567/ppandas/refs/heads/master/tests/data/numerical-2.csv")

    pd1 = PDataFrame(["Gender","Age"],df1)
    pd2 = PDataFrame(["Gender","Age"],df2)

    pd_join = pd1.pjoin(pd2,mismatches={"Age":'categorical'})

    queryResult = pd_join.query(['Gun Control'],{"Gender":'female',"Age":'[40,60)'})
    print('conditional query')
    print(queryResult)

def test_numerical():
    df1 = pd.read_csv("https://raw.githubusercontent.com/ecorbin567/ppandas/refs/heads/master/tests/data/numerical-1.csv")
    df2 = pd.read_csv("https://raw.githubusercontent.com/ecorbin567/ppandas/refs/heads/master/tests/data/numerical-2.csv")

    pd1 = PDataFrame(["Gender","Age"],df1)
    pd2 = PDataFrame(["Gender","Age"],df2)

    pd_join = pd1.pjoin(pd2,mismatches={"Age":'numerical'})

    queryResult = pd_join.query(['Gun Control'],{"Gender":'female',"Age":'[40,60)'})
    print('conditional query')
    print(queryResult)