exit()

#%%
#! QA (roberta-base-squad2)
from transformers import pipeline
pipe = pipeline(model="deepset/roberta-base-squad2") # ModelForQuestionAnswering 
pipe(question="Where do I live?", context="My name is Wolfgang and I live in Berlin")
# {'score': 0.9191, 'start': 34, 'end': 40, 'answer': 'Berlin'}



#%%
#! Summarization (BART)
from transformers import pipeline
pipe = pipeline("summarization") # ModelForConditionalGeneration (BartForConditionalGeneration)
print(type(pipe.model))
pipe("An apple a day, keeps the doctor away", min_length=5, max_length=20)



#%%
#! Table Question Answering
from transformers import pipeline, AutoTokenizer
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")

pipe = pipeline(model="google/tapas-base-finetuned-wtq") # ModelForTableQuestionAnswering
table = {
    "Repository": ["Transformers", "Datasets", "Tokenizers"],
    "Stars": ["36542", "4512", "3934"],
    "Contributors": ["651", "77", "34"],
    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
}
tokens = tokenizer(pd.DataFrame(table), return_tensors="pt")
print([tokenizer.convert_ids_to_tokens(x) for x in tokens.input_ids])
# [['[CLS]', '[SEP]', 'repository', 'stars', 'contributors', 'programming', 'language', 'transformers', '365', '##42', '65', '##1', 'python', 'data', '##set', '##s', '451', '##2', '77', 'python', 'token', '##izer', '##s', '39', '##34', '34', 'rust', ',', 'python', 'and', 'node', '##js']]
"""
    {'input_ids': tensor([[101, 102, 22409, 3340, 16884, 4730, 2653, 19081, 19342, 20958, 3515,  2487, 18750,  2951, 13462,  2015, 28161,  2475,  6255, 18750, 19204, 17629,  2015,  4464, 22022,  4090, 18399,  1010, 18750,  1998, 13045, 22578]]), 
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 
    'token_type_ids': tensor([[
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 2, 0, 0, 0, 0, 0],
            [1, 3, 0, 0, 0, 0, 0],
            [1, 4, 0, 0, 0, 0, 0],
            [1, 4, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 2, 1, 0, 3, 1, 0],
            [1, 2, 1, 0, 3, 1, 0],
            [1, 3, 1, 0, 3, 1, 0],
            [1, 3, 1, 0, 3, 1, 0],
            [1, 4, 1, 0, 0, 0, 0],
            [1, 1, 2, 0, 0, 0, 0],
            [1, 1, 2, 0, 0, 0, 0],
            [1, 1, 2, 0, 0, 0, 0],
            [1, 2, 2, 0, 2, 2, 0],
            [1, 2, 2, 0, 2, 2, 0],
            [1, 3, 2, 0, 2, 2, 0],
            [1, 4, 2, 0, 0, 0, 0],
            [1, 1, 3, 0, 0, 0, 0],
            ...
            [1, 4, 3, 0, 0, 0, 0],
            [1, 4, 3, 0, 0, 0, 0],
            [1, 4, 3, 0, 0, 0, 0],
            [1, 4, 3, 0, 0, 0, 0]]])
    }
"""

pipe(query="How many stars does the transformers repository have?", table=table)
# {'answer': 'AVERAGE > 36542', 'coordinates': [(0, 1)], 'cells': ['36542'], 'aggregator': 'AVERAGE'}


#%%
#! Text Classification Pipeline
from transformers import pipeline
pipe = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english") # ModelForSequenceClassification (DistilBertForSequenceClassification)
print(type(pipe.model))
pipe("This movie is disgustingly good !")
# [{'label': 'POSITIVE', 'score': 1.0}]

pipe("Director tried too much.")
# [{'label': 'NEGATIVE', 'score': 0.996}]



#%%
#! Text Generation Pipeline
from transformers import pipeline, set_seed
set_seed(0)
pipe = pipeline(model="openai-community/gpt2") # ModelWithLMHead
print(type(pipe.model))
pipe("I can't believe you did such a ", do_sample=False)
# [{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

# These parameters will return suggestions, and only the newly created text making it easier for prompting suggestions.
outputs = pipe("My tart needs some", num_return_sequences=4, return_full_text=False)


pipe = pipeline(model="HuggingFaceH4/zephyr-7b-beta")
# Zephyr-beta is a conversational model, so let's pass it a chat instead of a single string
pipe([{"role": "user", "content": "What is the capital of France? Answer in one word."}], do_sample=False, max_new_tokens=2)
# [{'generated_text': [{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}, {'role': 'assistant', 'content': 'Paris'}]}]


#%%
#! Token Classification Pipeline
from transformers import pipeline
pipe = pipeline(model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple") # ModelForTokenClassification
sentence = "Je m'appelle jean-baptiste et je vis à montréal"
tokens = pipe(sentence)
#[{'entity_group': 'PER', 'score': 0.9931, 'word': 'jean-baptiste', 'start': 12, 'end': 26}, {'entity_group': 'LOC', 'score': 0.998, 'word': 'montréal', 'start': 38, 'end': 47}]

token = tokens[0]
sentence[token["start"] : token["end"]] # Start and end provide an easy way to highlight words in the original text.


# Some models use the same idea to do part of speech.
pipe = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")
pipe("My name is Sarah and I live in London")
# [{'entity_group': 'PRON', 'score': 0.999, 'word': 'my', 'start': 0, 'end': 2}, {'entity_group': 'NOUN', 'score': 0.997, 'word': 'name', 'start': 3, 'end': 7}, {'entity_group': 'AUX', 'score': 0.994, 'word': 'is', 'start': 8, 'end': 10}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'sarah', 'start': 11, 'end': 16}, {'entity_group': 'CCONJ', 'score': 0.999, 'word': 'and', 'start': 17, 'end': 20}, {'entity_group': 'PRON', 'score': 0.999, 'word': 'i', 'start': 21, 'end': 22}, {'entity_group': 'VERB', 'score': 0.998, 'word': 'live', 'start': 23, 'end': 27}, {'entity_group': 'ADP', 'score': 0.999, 'word': 'in', 'start': 28, 'end': 30}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'london', 'start': 31, 'end': 37}]



#! Zero-Shot Classification Pipeline
from transformers import pipeline
pipe = pipeline(model="facebook/bart-large-mnli") # ModelForSequenceClassification
pipe(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)
# {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

pipe(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["english", "german"],
)
# {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['english', 'german'], 'scores': [0.814, 0.186]}


#%%
#! Text2Text Generation Pipeline
from transformers import pipeline
pipe = pipeline(model="mrm8488/t5-base-finetuned-question-generation-ap") # ModelForConditionalGeneration(T5ForConditionalGeneration)
print(type(pipe.model))
pipe("answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google")
# [{'generated_text': 'question: Who created the RuPERTa-base?'}]