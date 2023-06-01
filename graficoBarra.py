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
        ["Subjective",0.940676,0.942962,0.940676,0.945418,0.947721],
        ["Objective",0.951067,0.952198,0.950217,0.954995,0.956687],
        ["Assessment",0.689483,0.713258,0.713879,0.721006,0.732304],
        ["Plan",0.924350,0.926267,0.926360,0.932831,0.938170],
        ["Total Weighted",0.935094,0.937521,0.935750,0.940810,0.943571],
        ["Total Macro",0.876394,0.883671,0.882783,0.888562,0.893721]
                   ],
                  columns=['Model', 'MBERT','DistilBERT', 'BioBERT',  'BioBERTpt',  'BioBERTptRT'])


#df[['MBERT','BioBERT','DistilBERT']].plot(kind='bar')
#df['BioBERTpt'].plot(secondary_y=True)


# view data
print(df)
# plot grouped bar chart
ax= df.plot(x='Model',
        kind='bar',
        stacked=False,
        ylim=[0.6,1],
         
        title='F1-Score by class')


        
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) 



plt.savefig("fig1.pdf", bbox_inches='tight')

