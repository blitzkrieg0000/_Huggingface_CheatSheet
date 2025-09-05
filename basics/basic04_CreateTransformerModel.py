from transformers import BertConfig, BertModel


config = BertConfig(
    hidden_size=768,
    intermediate_size=3072,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=12
)


model = BertModel(config)
model = BertModel.from_pretrained("bert-base-cased")









