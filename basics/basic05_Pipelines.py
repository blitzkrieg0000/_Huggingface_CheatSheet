import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

#! Pipeline
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=torch.device("cuda:0"))
# pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=torch.device("cuda:0"))


results = pipe("I am having a bad day")
print(results)