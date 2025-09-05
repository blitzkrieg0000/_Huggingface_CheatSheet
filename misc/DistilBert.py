import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Embedding Katmanı (Word + Position Embedding)
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, max_seq_len=512):
        super(EmbeddingLayer, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        word_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(positions)
        embeddings = word_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# 2. Multi-Head Attention Katmanı
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)
        
        # Linear transformations for query, key, value
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim) # [1, 4, 768] => [1, 4, 12, 64]
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose to get dimensions [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2) # [1, 12, 4, 64]
        K = K.transpose(1, 2) 
        V = V.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5  # Scaled dot-product attention
        
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax for attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Attention output
        output = torch.matmul(attn_weights, V)
        
        # Concat and pass through the output linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_linear(output)
        return output, attn_weights


# 3. Transformer Block (Self-Attention + Feed Forward Network)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.attn_layer_norm = nn.LayerNorm(embed_size, eps=1e-12)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.GELU(),
            nn.Linear(ff_hidden_size, embed_size)
        )
        self.ffn_layer_norm = nn.LayerNorm(embed_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None):
        # Self-Attention
        attention_output, _ = self.attention(x, x, x, attention_mask)
        attention_output = self.attn_layer_norm(attention_output + x)
        
        # Feed Forward Network
        ffn_output = self.ffn(attention_output)
        ffn_output = self.ffn_layer_norm(ffn_output + attention_output)
        return self.dropout(ffn_output)


# 4. Model (Transformer)
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden_size):
        super(TransformerModel, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_size)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_hidden_size) for _ in range(num_layers)
        ])
    
    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        for block in self.encoder_blocks:
            x = block(x, attention_mask)
        return x



# 5. Test
vocab_size = 30522  # BERT vocab size
embed_size = 768    # BERT hidden size
num_heads = 12      # BERT num heads
num_layers = 6      # DistilBERT uses 6 layers
ff_hidden_size = 3072

model = TransformerModel(vocab_size, embed_size, num_heads, num_layers, ff_hidden_size)


input_ids = torch.tensor([[101, 2023, 2789, 102]])
outputs = model(input_ids)

print(outputs.shape)
