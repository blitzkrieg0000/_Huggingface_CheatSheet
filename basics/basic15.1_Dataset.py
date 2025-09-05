# wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
"""
    ! Huggingface dataset kütüphanesinin kullanımı
""" 
import html

from datasets import load_dataset, Dataset
import pandas as pd
from rich import print
from transformers import AutoTokenizer

# =================================================================================================================== #
#! dataset ile CSV Verilerini Yükle
# =================================================================================================================== #
data_files = {
    "train": "temp/dataset/drug/drugsComTrain_raw.tsv",
    "test": "temp/dataset/drug/drugsComTest_raw.tsv"
}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")


# İlk 1000 satırı al
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# print(drug_sample[:3])


# Adsız sütunları yeniden adlandır
drug_dataset = drug_dataset.rename_column("Unnamed: 0", "patient_id")


# None satıra sahip olmayan "condition" satırlarını al (None'lardan kurtul)
drug_dataset = drug_dataset.filter(lambda example: example["condition"] is not None)


# MAP özelliği ile küçük harfe döndürme
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

drug_dataset = drug_dataset.map(lowercase_condition)
print("Conditions: ", drug_dataset["train"]["condition"][:3])


# Yeni bir sütun oluştur
# dataset.add_column(<array>)
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset_review = drug_dataset.map(compute_review_length)
print("Yeni Training Veriseti:", drug_dataset_review["train"][0])


# Sort
drug_dataset_review["train"] = drug_dataset_review["train"].sort("review_length")
print("Yeni Training Veriseti:", drug_dataset_review["train"][0])


# Eğer review 30 kelimeden küçükse dikkate alma
drug_dataset_review = drug_dataset_review.filter(lambda x: x["review_length"] > 30)
print(drug_dataset_review.num_rows)


# Review sütunundaki html karakterleri kaldır.
drug_dataset_review = drug_dataset_review.map(lambda x: {"review": html.unescape(x["review"])})


# Review sütununu batch olarak ele alma
new_drug_dataset_review = drug_dataset_review.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, 
    batched=True,
    num_proc=8
)




# =================================================================================================================== #
#! TOKENIZER ile dataset Kullanımı
# =================================================================================================================== #
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)

# Test
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True
    )
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

"""
    => Eğer "datasets" ile oluşturulan bir verisetinde "batch" halinde bir mapping çalıştırırsak, 
    geriye batch ile aynı boyutta bir yapı döndürmemiz gerekiyor.

    => Burada "drug_dataset.map()" metodunun aldığı "tokenize_and_split" fonksiyonu içerisinde 
    tokenizerda kullanılan argümanlar dolayı "input_ids" için taşan tokenlarıda bir liste olarak
    döndürüyor.

    => Örneğin batch boyutu 2 olduğunda tek bir kolon için [[1, 2, 3], [4, 5, 6]] şeklinde geriye dönen ifade olduğunda 
    sorun çıkmazken, taşan tokenlarda array boyutu düzgün olmadığı için diğer verilerin 
    tekrarlanması gerekiyor. Şu şekilde bir {"input_ids": [[1, 2], [3], [4], [5]]} geri dönüş olduğunda
    geriye 2 elemanlı bir batch beklenirken 4 elemanlı olmasından dolayı ve hata alıyoruz.

    => Çözüm olarak eski kolonları kaldırabiliriz ancak bu seferde taşan tokenları almamış oluruz.
""" 
print("Column names: ", drug_dataset["train"].column_names)
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, 
    batched=True, 
    # remove_columns=drug_dataset["train"].column_names
)

traindata = tokenized_dataset["train"]






# =================================================================================================================== #
#! dataset Objesini Diğer Formatlarda Kullanma
# =================================================================================================================== #
#! Dataset objesi artık PANDAS veri tipinde veri döndürecektir. Tipi değişmeyecektir.
drug_dataset.set_format("pandas")


train_df: pd.DataFrame = drug_dataset["train"][:]
print(type(train_df))
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"count": "frequency"})
)
print(frequencies.head())

new_dataset = Dataset.from_pandas(frequencies)
print("\nYeni Veriseti: ", new_dataset)



#! Dataset objesini Arrow'a sıfırla
drug_dataset.reset_format()


# SPLIT
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
drug_dataset_clean["test"] = drug_dataset["test"]
print(drug_dataset_clean)


# SAVE AS ARROW
drug_dataset_clean.save_to_disk("temp/dataset/drug/arrow")


# SAVE AS JSON
# for split, dataset in drug_dataset_clean.items():
#     dataset.to_json(f"drug-reviews-{split}.jsonl")


# LOAD FROM JSON
# data_files = {
#     "train": "drug-reviews-train.jsonl",
#     "validation": "drug-reviews-validation.jsonl",
#     "test": "drug-reviews-test.jsonl",
# }
# drug_dataset_reloaded = load_dataset("json", data_files=data_files)