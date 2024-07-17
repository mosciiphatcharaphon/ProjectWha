import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch


df = pd.read_excel('C:\\Users\\mhewwha\\OneDrive\\Desktop\\ProjectWha\\AI\\data\\Gas1.xlsx')

label_mapping = {'AccessGasSystem': 0, 'IndividualAccountGas': 1, 'MeanAgency': 2, 'TypeGasLogin': 3, 'WhyGasData': 4, 'UseGasOper': 5, 'ProblemDGAGas': 6 , 'ProblemINDGas':7}
df['label'] = df['type'].map(label_mapping)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Custom Dataset






# แยกข้อมูลเป็น train/test
train_texts, val_texts, train_labels, val_labels = train_test_split(df['message'], df['type'], test_size=0.2)

# สร้าง Dataset
train_dataset = CustomDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len=128)
val_dataset = CustomDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len=128)

# ตั้งค่าการฝึกโมเดล
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ฝึกโมเดล
trainer.train()

# ประเมินโมเดล
trainer.evaluate()
