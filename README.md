# 🤗 Hugging Face Transformers Basics

This repository provides a **collection of practical scripts** to explore Hugging Face's Transformers library.  
It covers the essential workflows such as model loading, tokenization, pipelines, dataset handling, training, and semantic search.  

---

## 📚 Contents

### 🔹 Model Basics
- AutoModel & AutoModelForSequenceClassification  
- Saving and loading transformer models  
- Creating a custom transformer model  

### 🔹 Tokenization
- Introduction to Hugging Face Tokenizers  
- Advanced tokenization examples (custom vocab, batching)  

### 🔹 Pipelines
- Using prebuilt pipelines for inference  
- Pipeline variations with text classification, Q&A, etc.  
- Batched pipelines for performance  

### 🔹 Data Handling
- Working with Hugging Face Datasets  
- Streaming large datasets efficiently  
- Using Apache Arrow format with datasets  

### 🔹 Training
- First training loop example  
- Advanced training with custom loops  
- Accelerated training with `Accelerator`  
- Data preprocessing before training  

### 🔹 Generation
- Decoder–Generator models for text generation  
- Asymmetric Semantic Search examples  

### 🔹 Miscellaneous
- DistilBERT example usage  
- Audio processing scripts  
- Utility helpers  

---

## Tools
```bash
# Install dependencies
pip install transformers datasets accelerate
