from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertForSequenceClassification, AdamW #, BertConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences# Set the maximum sequence length.
from config import Lista as Lista
import sys
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import torch

MAX_LEN = 512


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

#modelName =  "pucpr/biobertpt-all"
seed= 42

modelName = sys.argv[1]
#modelName =  "bioBertPTRetreino"
config =  getattr(Lista,modelName)
if(not config):
    print("Erro configuracao nao encontrada:"+ sys.argv[1])
    os._exit(0)

if torch.cuda.is_available():
  print("\nUsing: ", torch.cuda.get_device_name(0))
  device = torch.device('cuda')
else:
    print('No GPU available, using the CPU instead.')
    exit()


torch.manual_seed(config.seed_val)
torch.cuda.manual_seed_all(config.seed_val)


df = pd.read_csv(config.file_labels, delimiter=',', header=None, names=['sentence','label'])# Report the number of sentences.

print('Number of training sentences: {:,}\n'.format(df.shape[0]))
output_dir='./result/'+modelName

output_logs = output_dir+"/logs"
if not os.path.exists(output_logs):
    os.makedirs(output_logs) 
output_dir_train = output_dir+"/train"

output_metric=output_dir+"/metric"
if not os.path.exists(output_metric+"/parcial"):
    os.makedirs(output_metric+"/parcial") 

n=5
X = df.sentence
y = df.label

skf = StratifiedKFold(n_splits=n, random_state=seed, shuffle=True)

results = []
cont=0

tokenizer = BertTokenizer.from_pretrained(config.modelBERT)

metric = evaluate.load("accuracy")

KFoldStep=0
EpochStep=0
IntimeSentence=""

def compute_metrics(eval_pred):
    global EpochStep,KFoldStep

    predictions, labels = eval_pred
    predictionsRaw=predictions
    predictions = np.argmax(predictions, axis=1)
    report=(classification_report(labels, predictions, digits=4,  output_dict=True))
    dfTmp = pd.DataFrame(report).transpose()
    dfTmp.to_csv(output_metric+'/parcial/report_kfold_'+str(KFoldStep)+'_epoch_'+str(EpochStep)+'.csv', header=True, index=True)

    outHtml=dfTmp.to_html( header=True, index=True)

    fileHtml = open(output_metric+'/parcial/report_kfold_'+str(KFoldStep)+'.html', "a+") 
    fileHtml.write("<br/><label>Epoch:"+str(EpochStep) +"</label><br/>")
    fileHtml.write(outHtml)
    fileHtml.close()

            #"Subjective","Objective","Assessment","Plan"
    resultado=[]
    for index, rotulo in enumerate(labels):
        softMax= np.exp(predictionsRaw[index]) / np.sum(np.exp(predictionsRaw[index]))   
        lista=[]
        for val in softMax:
            lista.append(str(round(val,ndigits=8)))
        resultado.append("|".join(lista))

    dfTmp2 = pd.DataFrame({"label":labels, "predictions":predictions, "resultado":resultado , "sentence":IntimeSentence})
    dfTmp2.to_csv(output_metric+'/parcial/result_kfold_'+str(KFoldStep)+'_epoch_'+str(EpochStep)+'.csv', header=False, index=False)

    if(EpochStep=="final"):
        dfTmp.to_csv(output_metric+'/report_kfold_'+str(KFoldStep)+'.csv', header=True, index=True)
        dfTmp2.to_csv(output_metric+'/result_kfold_'+str(KFoldStep)+'.csv', header=False, index=False)
        fileHtml = open(output_metric+'/report.html', "a+") 
        fileHtml.write("<br/><label>KFold:"+str(KFoldStep) +"</label><br/>")
        fileHtml.write(outHtml)
        fileHtml.close()

 
    if(EpochStep!="final"):
        EpochStep=EpochStep+1

    return metric.compute(predictions=predictions, references=labels)

class MyDataset:
    def __init__(self, text, label,tokenizer):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = ' '.join(self.text[item].split())
        label = self.label[item]
        enc = self.tokenizer(text, max_length=MAX_LEN, truncation=True, padding='max_length', return_tensors='pt')

        return {
            'input_ids': enc.input_ids[0],
            'attention_mask': enc.attention_mask[0],
            'token_type_ids': enc.token_type_ids[0],
            'labels': torch.tensor(label, dtype=torch.long)
        } 


for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    EpochStep=1
    KFoldStep=i
    model = AutoModelForSequenceClassification.from_pretrained(config.modelBERT, num_labels=4)
    model.to(device)

    IntimeSentence =test_df.sentence.values.tolist()
    X_train_tokenized = tokenizer(train_df.sentence.values.tolist(), padding=True, truncation=True, max_length=MAX_LEN)
    X_val_tokenized = tokenizer(test_df.sentence.values.tolist(), padding=True, truncation=True, max_length=MAX_LEN)

    train_dataset = MyDataset( 
            text=train_df.sentence.values,
            label=train_df.label.values,
            tokenizer=tokenizer)

    val_dataset = MyDataset( 
            text=test_df.sentence.values,
            label=test_df.label.values,
            tokenizer=tokenizer)

    training_args = TrainingArguments(
    output_dir=output_dir_train, 
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
	num_train_epochs=4,
    optim="adamw_torch_fused", 
    logging_dir=output_logs,
    logging_strategy="steps",
    logging_steps=200,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,

    )


    trainer.train()
    EpochStep="final"
    metrics=trainer.evaluate()

    print("metrics")
    print(metrics)

    output_dir_model = './result/'+modelName+"/model_save/k_"+str(KFoldStep)

    if not os.path.exists(output_dir_model):
        os.makedirs(output_dir_model) 

    print("Saving model to %s" % output_dir_model)
    model.save_pretrained(output_dir_model)
    tokenizer.save_pretrained(output_dir_model)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    #exit() 



