from transformers import pipeline

pipe = pipeline("fill-mask")   # distilbert/distilroberta-base

results = pipe(
    "This course will teach you all about <mask> models.",
    top_k=2
)

print(results)
"""
    [
        {'score': 0.19198468327522278, 'token': 30412, 'token_str': ' mathematical', 'sequence': 'This course will teach you all about mathematical models.'},
        {'score': 0.042092032730579376, 'token': 38163, 'token_str': ' computational', 'sequence': 'This course will teach you all about computational models.'}
    ]
"""