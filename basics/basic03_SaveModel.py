from rich import print
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification



model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


#! Model Save
pt_save_directory = "./weight/model/saved_model"
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)


#! Model Load
pt_model = AutoModelForSequenceClassification.from_pretrained("./weight/model/saved_model")
print(pt_model)