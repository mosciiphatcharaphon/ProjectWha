import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_excel('C:\\Users\\ritht\\Downloads\\ProjectWha\\AI\\data\\law.xlsx')

#clean data
def cleanText(text):    
    text = text.replace(" ","")
    return text
df["message"] = df["message"].apply(cleanText)

#แปลง label ให้เป็นเลข
Label_Encorder = LabelEncoder()
df["Label"] = Label_Encorder.fit_transform(df["type"])
print(df.head())
print(np.unique(df["Label"]))

#ทำเทรนและเทส
train_df , test_df = train_test_split(df,test_size=0.2,random_state=42)

trainDataframe = Dataset.from_pandas(train_df)
testDataframe = Dataset.from_pandas(test_df)

#ทำโมเดล
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)

#โหลดmodel
model = BertForSequenceClassification.from_pretrained(model_name,num_labels=len(df["Label"].unique()))

def tokenized_and_labels(examples):
    tokenized = tokenizer(examples["message"] , padding = "max_length" , truncation = True,Max_length = 512)
    tokenized["labels"] = examples["labels"]
    return tokenized


train_tokenized = trainDataframe.map(tokenized_and_labels,batched=True)
test_tokenized = testDataframe.map(tokenized_and_labels,batched=True)


def computer_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits,axis=-1)
    return{
        "accuracy" : accuracy_score(labels, predictions),
        "f1" : accuracy_score(labels,predictions),
    }

    



training_args = TrainingArguments(
    output_dir = './results',
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 64,
    warmup_steps = 500,
    weight_decay = 0.01,
    logging_dir = './logs',
    logging_steps = 10,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_tokenized,
    eval_dataset = test_tokenized,



)

trainer.train()

model.save_pretrained("./lawAi")
tokenizer.save_pretranied("./lawAi")




