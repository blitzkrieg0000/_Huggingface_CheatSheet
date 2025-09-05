from transformers import BertTokenizer
from Tool.Default import LogColorDefaults
from Tool.Core import ConsoleLog


ConsoleLog("Merhaba", LogColorDefaults.Success, True)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

tokenizer.save_pretrained("./weight/tokenizer/")

text = "Using a Transformer network is simple!"

tokenized_text = tokenizer(text, return_tensors="pt")
print("\033[1;31mInputs IDs: \033[0m", tokenized_text)



#! Adım adım yukarıdaki tokenizer işlemini açalım.
"""
        Bu tokenizer bir subword-tokenizerdır: Sözcük dağarcığı(vocabulary) tarafından temsil edilebilecek tokenler
    elde edene kadar sözcükleri böler. 
    
    Burada "transformer" iki tokena ayrılmıştır: "trans" ve "##former"
"""
tokens = tokenizer.tokenize(text)
print("\033[1;35mTokens: \033[0m", tokens)



"""
        Burada tokenleri Vocabularyde karşılık gelen ID'lere çeviriyoruz ancak bu metot
    bazı özel tokenleri (BOS, EOS, SEP, UNK vb.) eklemiyor.
"""
ids = tokenizer.convert_tokens_to_ids(tokens)
print("\033[1;35mInput IDs: \033[0m", ids)


# Decode
decoded_string = tokenizer.decode(ids)
print("\033[1;36mDecoded String: \033[0m", decoded_string)




