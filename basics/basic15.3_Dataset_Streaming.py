from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


en = load_dataset(
    "allenai/c4",
    "tr",
    cache_dir="temp/dataset/c4",
    streaming=True
)


train_data = en["train"].shuffle(buffer_size=100, seed=42)



dataset = DataLoader(
    train_data,
    batch_size=2,
    collate_fn=lambda x: x
)


for data in dataset:
    print(data)
    break