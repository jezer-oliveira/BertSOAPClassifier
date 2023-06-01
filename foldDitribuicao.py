from config import Lista as Lista
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

import seaborn as sns
#import cf_matrix as cf
from sklearn.metrics import classification_report

labelsLabel=[
            "Subjective",
            "Objective",
            "Assessment",
            "Plan"
        ]


df = pd.read_csv(f"saida/bioBertPTRetreino_3.csv", delimiter=',',header=None, names=["fold",'labels','predictions',"results", "sentence"] )
fold = df['fold'].tolist()
y_true = df['labels'].tolist()
folds=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
group=[0,0,0,0,0]
for x in range(len(fold)):
    folds[y_true[x]][fold[x]]=folds[y_true[x]][fold[x]]+1
    group[fold[x]]=group[fold[x]]+1
print(group)

