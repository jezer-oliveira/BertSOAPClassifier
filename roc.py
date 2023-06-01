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


from config import Lista as Lista


modelName =  "bioBertPTRetreino"
config =  getattr(Lista,modelName)

n_classes = config.num_labels

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score



df = pd.read_csv(f"saida/{modelName}_3.csv", delimiter=',',header=None, names=["fold",'labels','predictions',"results", "sentence"] )# Report the number of sentences.


y_true = df['labels'].tolist()
y_pred = df['predictions'].tolist()


y_true=[]
y_pred=[]
totalAcerto=0
total=0
results=[]
predics=[]

#for sentence, label in df.items():
#    print(f'label: {label}')
#    print(f'content: {sentence}', sep='\n')
#print(df)
for _,row in df.iterrows():

    value= int(row[1])
    predit= int(row[2])
    preditArray=[0]*config.num_labels
    preditFixArray=[0]*config.num_labels
    valueArray=[0]*config.num_labels
    preditArray=  list(map(float, row[3].split("|")))
    valueArray[value]=1
    results.append(valueArray)
    predics.append(preditArray)

Y_test = np.array(results)
y_score = np.array(predics)
#print(type(Y_test))
#print(type(results))
#print(results)
print("y_score[:, i]")
print(y_score)
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(config.num_labels):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.4f}'
      .format(average_precision["micro"]))
print("average_precision")
print(average_precision)

# %%
# Plot the micro-averaged Precision-Recall curve
# ...............................................
#

plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.4f}'
    .format(average_precision["micro"]))

plt.savefig("saida/imgs/"+modelName+"plot1_novo.pdf")

# %%
# Plot Precision-Recall curve for each class and iso-f1 curves
# .............................................................
#
from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.4f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(config.num_labels), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.4f})'
                  ''.format(config.labels[i], average_precision[i]))

    #labels.append('Precision-recall for class {0} (area = {1:0.4f})'
    #              ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.30)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall')
plt.legend(lines, labels, loc=(0, -.50), prop=dict(size=14))


plt.savefig("saida/imgs/"+modelName+"_plot2_novo.pdf")

#############################################
# Plot ROC curves for the multilabel problem
#############################################
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from scipy import interp

y_test = np.array(results)
y_score = np.array(predics)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

# %%
# Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[2])


for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(config.labels[i], roc_auc[i]))



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("saida/imgs/"+modelName+"plotroc_novo.pdf")

y_prob = predics

macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                  average="macro")
weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                     average="weighted")
macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                  average="macro")
weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                     average="weighted")
print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
print(config.out_dir)