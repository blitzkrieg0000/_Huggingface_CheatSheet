from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader


from Tool.Core import ColoredText
from Tool.Default import LogColorDefaults


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]


print(ColoredText("Veri Seti: ", LogColorDefaults.Error, True), raw_datasets)
print(ColoredText("Veri Seti Features: ", LogColorDefaults.Error, True), raw_datasets["train"].features)
print(ColoredText("Veri Seti Örnek 0: ", LogColorDefaults.Warning, True), raw_train_dataset[0])

# bert-base-uncased Tokenizer iki cümleyi birleştirerek kullanır. Sonraki Cümle Tahmini (NextSentencePrediction) için önemlidir.
sentence01 = raw_datasets["train"][0]["sentence1"]
sentence02 = raw_datasets["train"][0]["sentence2"]
tokenized_sentences = tokenizer(sentence01, sentence02, truncation=True)
print(ColoredText("İki cümle birleşmiş olarak çıktı: ", LogColorDefaults.Info), tokenized_sentences)

# Ayrıca her bir cümledeki tokenleri da [SEP] ile ayırarak "token_type_ids" lerini 0 ve 1 lerden oluşan bir maske olarak ayarlar.
ctokens = tokenizer.convert_ids_to_tokens(tokenized_sentences.input_ids)
print(ColoredText("\nBirleştirilmiş Tokenler: ", LogColorDefaults.Info), ctokens)


#! TOKEN MAPS
# "datasets" kütüphanesi verisetlerini "ApacheArrow" ile yükler, bu yüzden RAM kullanımını optimize için map ile işlem yapabiliriz.
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)  # num_proc=4
print(ColoredText("Tokenized Datasets: ", LogColorDefaults.Error, True), tokenized_datasets)
tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.with_format("torch")


#! Add Padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8
)

for step, batch in enumerate(train_dataloader):
    print(batch.input_ids.shape)
    break