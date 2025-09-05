import os

import jax
import numpy as np
import psutil
import pyarrow as pa
# from pyarrow import cuda  # pyarrow'u derlemek gerekiyor


def MemoryUsage():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def CreateDummyArrowFile(size_mb, file_path):
    # Veritabanı boyutunu byte cinsinden hesapla
    size_bytes = size_mb * 1024 * 1024
    
    bytes_per_item = np.dtype(np.int32).itemsize
    
    # Kaç adet eleman gerektiğini hesapla
    num_items = size_bytes // bytes_per_item
    
    # Rastgele veriler oluştur
    data = np.random.randint(low=0, high=1000, size=num_items, dtype=np.int32)
    
    
    ##! --------------- Arrow Table oluşturma --------------- !##
    table = pa.table(
        {"numbers": pa.array(data)
    })
    
    
    ##! --------------- Arrow dosyasını yaz --------------- !##
    with pa.OSFile(file_path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    
    # Dosyanın boyutunu kontrol et
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB cinsinden
    print(f"{file_path} başarıyla oluşturuldu. Dosya boyutu: {file_size:.2f} MB")
    
    return file_path


def ReadArrowIPC(file_path):
    with pa.ipc.open_file(file_path) as reader:
        arrow_table = reader.read_all()
    return arrow_table


def ReadArrowIPCWithMemoryMapping(file_path):
    dataset = None
    with pa.memory_map(file_path) as mmapped_file:
        with pa.ipc.open_file(mmapped_file) as reader:
            dataset = reader.read_all()

            # for i in range(reader.num_record_batches):
            #     dataset = reader.get_record_batch(i)

    return dataset


if "__main__" == __name__:
    print(f"Başlangıç Memory Kullanımı: {MemoryUsage(): .2f}")
    file_path = "temp/dataset/test/dummy.arrow"

    #! Save
    # file_path = CreateDummyArrowFile(4096, file_path)

    #! Load
    # dataset = ReadArrowIPC(file_path)
    arrowTable = ReadArrowIPCWithMemoryMapping(file_path)
    
    # ChunkedArray
    # Datasetteki her bir sütun ChunkedArray olarak tutulabilir:
    # for i in range(arrowTable["numbers"].num_chunks):
    #     for element in arrowTable["numbers"].chunk(i):
    #         print(element)


    # CUDA
    # ctx = cuda.Context(0)
    # cuda_buffer = ctx.buffer_from_data(arrowTable)
    # cpu_buffer = cuda_buffer.copy_to_host()


    # JAX ile CUDA'ya gönderme
    arrowArray = pa.array(arrowTable["numbers"][:100000])
    dlpackArray = jax.dlpack.from_dlpack(arrowArray, device=jax.devices("cuda")[0])
    summation = dlpackArray.sum()

    

    print(f"memory Kullanımı: {MemoryUsage(): .2f}")
    exit()
    #! Sadece bir adet kolonu okuyalım ve pandasa çevirelim
    numbers = arrowTable["numbers"]
    numbers = numbers.to_pandas()
    print(f"Bir adet kolon çevrildikten sonra memory Kullanımı: {MemoryUsage(): .2f}")


    
    #! Table'ın tamamını çevirelim
    pandasTable = arrowTable.to_pandas()
    print(type(pandasTable))
    # print("Table: ", numbers_pandas.head())
    print(f"Tablonun tamamı çevrildikten sonra memory Kullanımı: {MemoryUsage(): .2f}")
