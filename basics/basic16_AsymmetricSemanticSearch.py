"""
    PRETRAINED EMBEDDING MODELS
    # https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#
"""

import pandas as pd
from datasets import Dataset, load_dataset
from rich import print
from transformers import AutoModel, AutoTokenizer
import torch


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# LOAD DATASET
issues_dataset = load_dataset("lewtun/github-issues", split="train")
print(issues_dataset)



# FILTER
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)


# REMOVE COLUMNS
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"] # Bu sütunlar hariç kaldır...
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)

print(issues_dataset["comments"][0])


# DATASET => PANDAS
issues_dataset.set_format("pandas")
df:pd.DataFrame = issues_dataset[:]

# Her satırdaki liste şeklinde tutulan yorumları aynı row index'e sahip olarak ayır.
comments_df = df.explode("comments", ignore_index=True)

# PANDAS => DATASET
comments_dataset = Dataset.from_pandas(comments_df)

# Yorum uzunluklarını tutan kolonu ekle
comments_dataset = comments_dataset.map(lambda x: {"comment_length": len(x["comments"].split())})


# Yorum uzunlukları 15'ten büyükse dikkate al
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)


# Tüm içerikleri birleştir ve "text" kolonuna ata
def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }

comments_dataset = comments_dataset.map(concatenate_text)




# =================================================================================================================== #
#! EMBEDDING Modeli
# =================================================================================================================== #
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
model = model.to(DEVICE)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


embedding = get_embeddings(comments_dataset["text"][0])
print("Tek bir cümle için embedding boyutu: ", embedding.shape)


##! --------------- Tüm Veriseti için Embeddingleri Hesapla --------------- !##
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)


# FAISS (Facebook AI Similarity Search)
embeddings_dataset.add_faiss_index(column="embeddings")


# Query
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape


# Search
scores, samples = embeddings_dataset.get_nearest_examples("embeddings", question_embedding, k=5)


# TO PANDAS
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)


# SHOW
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50, "\n")
