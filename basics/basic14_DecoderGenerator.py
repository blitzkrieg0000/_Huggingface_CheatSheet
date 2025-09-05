import time
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Model ve tokenizer'ı yükleyin
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Test cümlesi
input_text = "Translate English to French: How are you?"

# Tokenize input
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Encoder kısmını çalıştırın
encoder_outputs = model.encoder(input_ids)  # Encoder çıktısı
encoder_hidden_states = encoder_outputs.last_hidden_state  # Encoder'ın ara temsili

# Decoder kısmını bir generator gibi adım adım çalıştırma
decoder_input_ids = torch.tensor([[model.config.pad_token_id]])  # İlk token (başlangıç için padding)
generated_ids = decoder_input_ids.clone()  # Sonuçları buraya toplayacağız
max_length = 50  # Maksimum token uzunluğu
num_tokens_generated = 0

# Başlangıç zamanını kaydedin
start_time = time.time()

while num_tokens_generated < max_length:
    # Decoder'ı bir adım çalıştır (encoder çıktısını ve decoder inputlarını vererek)
    decoder_outputs = model(
        input_ids=input_ids,  # Encoder girişi
        decoder_input_ids=decoder_input_ids,  # Decoder girişini veriyoruz
        encoder_outputs=encoder_outputs,  # Encoder çıktısını veriyoruz
        use_cache=True  # Performans için cache kullanımı
    )
    
    # Çıktıdan bir token al
    next_token_logits = decoder_outputs.logits[:, -1, :]  # Son token'ın logits değerleri
    next_token_id = torch.argmax(next_token_logits, dim=-1)  # En yüksek olasılıkla token'ı seç

    # Yeni token'ı ekle
    decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(-1)], dim=-1)
    generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)
    print(generated_ids)
    num_tokens_generated += 1
    
    # Eğer EOS token'ına ulaşıldıysa dur
    if next_token_id.item() == model.config.eos_token_id:
        break

# Çıktıyı çözümle ve yazdır
decoded_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
decoded_words = [tokenizer.decode([token_id]) for token_id in generated_ids[0]]

end_time = time.time()
total_time = end_time - start_time
tokens = len(generated_ids[0])

itlt = total_time / tokens  # ITL hesaplama

# Çıktıyı yazdırın
print(f"Decoded tokens: {decoded_tokens}")
print(f"Decoded words: {decoded_words}")
print(f"Total inference time: {total_time:.4f} seconds")
print(f"ITL per token: {itlt:.4f} seconds")
