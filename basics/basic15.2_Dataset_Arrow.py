"""
    ! Arrow in-memory veri formatının memory kullanımını düşürmedeki etkisi adına  yapılmış bir çalışma
    
    => dataset kütüphanesinin kullandığı arrow formatının apache arrow ile uyumsuzluğunun görüntülenmesi. 
    dataset kütüphanesi ile kaydedilen arrow formatlı dosyalar, apache arrow ile okunamamaktadır çünkü HF
    dataset kütüphanesi arrow formatını özelleştirerek kullanmaktadır.
"""

import io

import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from rich import print

print(f"Başlangıç Memory Kullanımı: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")


def MemoryUsage():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def ReadWithPandas(data_files):
    dataset = pd.read_json(data_files, lines=True)
    return dataset


def ReadWithDataset(data_files):
    dataset = load_dataset(
        "json", 
        data_files=data_files,
        # streaming=True
    )
    print(type(dataset["train"]))
    return dataset


def ReadWithArrowIPC(arrow_file):
    mmapped_table = pa.memory_map(arrow_file, "r")

    # Dosyanın açılması
    with pa.ipc.open_file(mmapped_table) as reader:
        dataset = reader.read_all()
    print(type(dataset))
    return dataset


def SaveDatasetAsArrow(dataset, parrow_file):
    buffer = io.BytesIO()
    dataset.to_parquet(buffer)
    arrow_table = pq.read_table(buffer)

    with pa.ipc.new_file(parrow_file, arrow_table.schema) as writer:
        writer.write(arrow_table)




if "__main__" == __name__:
    # Orijinal veri
    data_files = "temp/dataset/code/data_1021_time1626314310_default.jsonl"

    # huggingface ile oluşturulan arrow dosyası
    darrow_file = "temp/dataset/code/arrow/data-00000-of-00001.arrow"
    
    # pyarrow ile oluşturulan arrow dosyası
    parrow_file = "temp/dataset/code/parrow/data.arrow"


    #! Dataset
    print(f"\nHuggingface dataset yüklenmeden önce Memory Kullanımı: {MemoryUsage(): .3f} MB")
    dataset = ReadWithDataset(data_files)
    # SaveDatasetAsArrow(dataset, parrow_file)
    # dataset.save_to_disk("temp/dataset/code/arrow/")
    print(f"Huggingface dataset yüklendi. Memory Kullanımı: {MemoryUsage(): .3f} MB")


    #! Pandas
    print(f"\nPandas veriseti yüklenmeden önce Memory Kullanımı: {MemoryUsage(): .3f} MB")
    dataset2 = ReadWithPandas(data_files)
    print(f"Pandas veriseti yüklendi. Memory Kullanımı: {MemoryUsage(): .3f} MB")


    #! Arrow
    print(f"\nArrow veriseti yüklenmeden önce Memory Kullanımı: {MemoryUsage(): .3f} MB")
    dataset3 = ReadWithArrowIPC(parrow_file)
    print(f"Arrow veriseti yüklendi. Memory Kullanımı: {MemoryUsage(): .3f} MB")



