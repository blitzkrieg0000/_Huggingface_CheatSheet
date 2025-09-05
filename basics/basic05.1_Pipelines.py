import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline


pipe = pipeline("zero-shot-classification", device=0)


## Pipeline kullanırken zero-shot learning için etiketler verilebilir.
results = pipe(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)


print(results) # Çıktı: {'sequence': 'This is a course about the Transformers library', 'labels': ['education', 'business', 'politics'], 'scores': [0.8445960879325867, 0.11197637021541595, 0.04342753440141678]}