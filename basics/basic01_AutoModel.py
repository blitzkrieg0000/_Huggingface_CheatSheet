import warnings
warnings.filterwarnings("ignore")
from rich import print

from transformers import AutoTokenizer, AutoModel


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
print(model)


raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life. And I can't wait to get started!",
    "I hate this so much!",
    "A",
    "AAA",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=False, return_tensors="pt")

print("BOS Token: ", tokenizer.bos_token)
print("EOS Token: ", tokenizer.eos_token)
print("PAD Token: ", tokenizer.pad_token, tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
print("CLS Token: ", tokenizer.cls_token, tokenizer.convert_tokens_to_ids(tokenizer.cls_token))
print("SEP Token: ", tokenizer.sep_token, tokenizer.convert_tokens_to_ids(tokenizer.sep_token))
print("UNK Token: ", tokenizer.unk_token)
print("13360 Token: ", tokenizer.convert_ids_to_tokens(13360))


print("Input Token: \n", inputs.input_ids)
print("Attention Mask: \n", inputs.attention_mask)


for x in inputs.input_ids:
    print(tokenizer.decode(x))


outputs = model(**inputs)
print(outputs[0].shape)
print(outputs.last_hidden_state.shape)

