import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_scheduler)
"""
    Accelerator, birden fazla GPU üzerinde dağıtık eğitimi koalylaştırmak için kullanılır.
    Verisetini dağıtır ve verileri GPU'lardaki modele yönlendirir.
    Tüm GPU'lardaki gradientlerin ortalamasına göre her GPU'daki modelin ağırlığını günceller.
    Pytorch'un DDP (Distributed Data Parallel) modunu kullanır.
"""

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# DATASET
raw_datasets = load_dataset("glue", "mrpc")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator)



#! HF ACCELERATOR------------------------------------------------------------------------------------------------------
accelerator = Accelerator()

checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)
train_dl, eval_dl, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)


num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler( "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps,)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        optimizer.zero_grad()

        #! Forward + Loss
        outputs = model(**batch)
        loss = outputs.loss

        #! Accelerator Backward
        accelerator.backward(loss)

        #! Optimize
        optimizer.step()
        lr_scheduler.step()
        
        progress_bar.update(1)





#? $ accelerate config
#? $ accelerate launch train.py