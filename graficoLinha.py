from config import Lista as Lista
import sys
import os
import pandas as pd
import numpy as np

#lista = ["bertBaseMultilingual2","distilbert","biobert2","bioBertPT"]
lista = ["bertBaseMultilingual","distilbert","biobert","bioBertPT"]
#lista = ["bertBaseMultilingual","distilbert","biobert","bioBertPT"]

import matplotlib.pyplot as plt
import pandas as pd


# create data
df = pd.DataFrame([
        #['Subjective', 0.941170, 0.924567, 0.937891, 0.945270], 
        ["Subjective",0.939837,0.943629,0.945535,0.944101,0.948139],
        #['Objective', 0.955925, 0.948555, 0.954913, 0.957659], 
        ["Objective",0.954503,0.954359,0.946468,0.955602,0.958637],
        #['Assessment', 0.723636, 0.674545, 0.729091, 0.756364],
        ["Assessment",0.651763,0.684478,0.683388,0.688113,0.707016],
        #['Plan', 0.953924, 0.919726, 0.945284, 0.957883]
        ["Plan",0.928942,0.926134,0.926494,0.944348,0.938373]
        ],

        columns=['Model', 'MBERT', 'DistilBERT', 'BioBERT', 'BioBERTpt', 'BioBERTptRT'])


#df[['MBERT','BioBERT','DistilBERT']].plot(kind='bar')
#df['BioBERTpt'].plot(secondary_y=True)


# view data
print(df)
# plot grouped bar chart
ax= df.plot(x='Model',
        kind='line',
        stacked=False,
        ylim=[0.6,1],
         
        title='Accuracy by class')


        
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) 



plt.savefig("fig5.pdf", bbox_inches='tight')

