import warnings

from tqdm import tqdm
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

#! Pipeline
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=torch.device("cuda:0"))
# pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=torch.device("cuda:0"))


class MyDataset(Dataset):
    def __init__(self):
        self.Inputs = [
            "I am having a bad day",
            "I am having a good day",
            "I am having a awesome day",
            "I have a exam result F",
            "I have a exam result S"
        ]

    def __len__(self):
        return 256

    def __getitem__(self, i):
        return self.Inputs[i % self.Inputs.__len__()]



dataset = MyDataset()

for batch_size in [64, 128]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        print(out)