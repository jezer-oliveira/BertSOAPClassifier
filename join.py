from config import Lista as Lista
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from os.path import exists



redes= ["distilbert","bertBaseMultilingual","biobert","bioBertPT", "bioBertPTRetreino"]
kfolds=[0,1,2,3,4]
epochs= [2,3,4]

for rede in redes:
    for epoch in epochs:
        f1= open(f'./saida/{rede}_{epoch-1}.csv', 'w')
        for kfold in kfolds:
            arquivo=f"./metricaServer/{rede}/parcial/result_kfold_{kfold}_epoch_{epoch}.csv" 
            #/metricaServer/bertBaseMultilingual/parcial/
            linhas = open(arquivo,'r').readlines()
            for linha in linhas:
                f1.write(str(kfold)+","+linha) #.replace("|",",")
        f1.close()
            
            

    
