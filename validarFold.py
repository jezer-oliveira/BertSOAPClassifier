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
import seaborn as sns
#import cf_matrix as cf
from sklearn.metrics import classification_report

labelsLabel=[
            "Subjective",
            "Objective",
            "Assessment",
            "Plan"
        ]


def validar(rede, epoca): 
    global labelsLabel
    df = pd.read_csv(f"saida/{rede}_{epoca}.csv", delimiter=',',header=None, names=["fold",'labels','predictions',"results", "sentence"] )# Report the number of sentences.
    for fold in range(5): 
        foldAtual = df.loc[lambda x: (x['fold']==fold)]
        for soap in range(4):
            itens = foldAtual.loc[lambda x: (x['labels']==soap)]
            acertos = itens.loc[lambda x: (x['labels']==x['predictions'])]
            print(rede+","+str(fold)+","+labelsLabel[soap]+","+(str(len(acertos)/len(itens))) )

            



validar("distilbert",3)
validar("bertBaseMultilingual",3)
validar("biobert",3)
validar("bioBertPT",3)
validar("bioBertPTRetreino",3)

