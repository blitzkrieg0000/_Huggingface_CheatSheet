import warnings
warnings.filterwarnings("ignore")
from rich import print

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
print(model)


raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
]


# TOKENIZER
inputs = tokenizer(raw_inputs, padding=True, truncation=False, return_tensors="pt")
print(tokenizer.bos_token, tokenizer.eos_token)
print(inputs.input_ids)
print(inputs.attention_mask)


for x in inputs.input_ids:
    print(tokenizer.decode(x))

# MODEL
outputs = model(**inputs)
print(outputs[0].shape)
print(outputs.logits.shape)


predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
results = torch.argmax(predictions, dim=-1)
print(results)

labels = model.config.id2label
print([labels[label.item()] for label in results])