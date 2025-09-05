from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


sequences = ["Using a Transformer network is simple!", "Transformers are awesome!"]

model_inputs = tokenizer(
    sequences,
    return_tensors="pt", 
    truncation=True,
    padding=True,
    max_length=3
)

print(model_inputs)
