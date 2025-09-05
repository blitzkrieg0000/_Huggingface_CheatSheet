from transformers import pipeline


pipe = pipeline("text-generation", model="distilgpt2", device=0)

results = pipe(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2
)

print(results)