import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from Tool.Core import ColoredText
from Tool.Default import LogColorDefaults

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "Dream big, work hard, and let your passion light the path to success!"

# Birden fazla cümle varsa PADDING gereklidir ve tokenizerda padding token için:
print(ColoredText("Tokenizer PAD Token Id: ", LogColorDefaults.Warning), tokenizer.pad_token_id)


tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)


# 1 Batch
input_ids = torch.tensor([ids])
print(input_ids.shape)


results = model(input_ids)
print(results)


# 2 Batch
input_ids = torch.tensor([ids, ids])
print(input_ids.shape)

results = model(input_ids)
print(results)



# 2 Batch
input_ids = torch.tensor([ids, ids])
print(input_ids.shape)


attentions = torch.ones_like(input_ids)
results = model(input_ids, attention_mask=attentions)
print(results)


# Attention MASKS
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)