

# Mini-BPE-Algorithm

A from-scratch **Byte Pair Encoding (BPE)** tokenizer project built for learning, experimentation, and intuition.

This repository implements the core ideas behind GPT-style byte-level tokenization:

- UTF-8 byte fallback
- BPE merge training
- Vocabulary construction
- Encode / decode pipeline
- Regex-based pre-tokenization
- JSON save/load cache system

> This project was built for educational purposes and conceptual understanding.  
> It is **not** intended to be a production-grade tokenizer like `tiktoken`, SentencePiece, or industrial Rust/C++ tokenization systems.

---

# Why This Project Matters

Most people use tokenizers as black boxes.

While this project helped me understand:

- How text becomes token IDs
- Why byte-level tokenization handles Unicode well
- How frequent subwords are learned
- Why tokenization improves compression
- How LLM pipelines preprocess language

---

# Features

## Core Functionality

- Train BPE merges on custom text data
- Build vocabulary dynamically
- Encode text into token IDs
- Decode token IDs back into text
- Save vocab and merges to disk
- Reload trained tokenizer instantly

## Byte-Level Robustness

Works with:

- English
- Hindi
- Emojis
- Mixed-language text
- Symbols / punctuation
- Arbitrary Unicode via UTF-8 bytes


---

# Project Structure

```text
Mini-BPE-Algorithm/
│── tokenizer.py          # Main tokenizer implementation
│── example1.py           # Training + encode/decode demo
│── example2.py           # Unicode / multilingual tests
│── example3.py           # Compression experiment on Python code
│── pyproject.toml        # Project dependencies / requirements
│
├── cache/
│   ├── vocab.json        # Saved vocabulary
│   └── merges.json       # Learned merge rules
│
└── data/
    └── *.txt             # Training corpus files

```

----------

# How It Works

## Step 1: Read Dataset

All `.txt` files inside the `data/` folder are loaded.

## Step 2: Regex Pre-tokenization

Text is split into chunks using a GPT4-style regex pattern.

## Step 3: Convert to UTF-8 Bytes

Each token becomes raw byte values.

Example:

```text
hello -> [104, 101, 108, 108, 111]

```

## Step 4: Learn Frequent Pairs

Most common adjacent byte/token pairs are merged repeatedly.

Example:

```text
(h, e) -> new token
(he, l) -> new token

```

## Step 5: Build Vocabulary

Base tokens:

```text
0 ... 255

```

represent raw bytes.

Merged tokens get new IDs above that range.

## Step 6: Encode / Decode

-   Encode text -> token IDs
    
-   Decode token IDs -> original text
    

Round-trip correctness is preserved.

----------

# Example Files

# example1.py — Full Training Demo

Shows:

-   Train tokenizer on `data/`
    
-   Save vocab + merges
    
-   Encode custom text
    
-   Decode back
    
-   Verify correctness
    

Example output includes multilingual text such as:

```text
Hello world! नमस्ते
```

----------

# example2.py — Robustness Tests

Tests multiple inputs such as:

-   lowercase text
    
-   uppercase text
    
-   Hindi
    
-   emoji
    
-   numbers
    
-   mixed-language strings
    
Useful for validating Unicode safety and reversibility.

----------

# example3.py — Compression Demo

Encodes a large Python code sample and compares:

-   original character length
    
-   tokenized length
    
-   percentage reduction
    
Observed result:
```text
33.12% reduction
```
Even when the tokenizer was not specifically trained on Python code.
This demonstrates that learned merges often generalize to structured text.

----------

# Usage

### Train Tokenizer

```python
from tokenizer import BPE

tokenizer = BPE(vocab_size=30000, n_merges=3000)
tokenizer.train("data")

```

This creates:
-   `vocab.json`
    
-   `merges.json`
    

----------

### Encode Text

```python
ids = tokenizer.encoder("Hello world!")
print(ids)

```

----------

### Decode Text

```python
text = tokenizer.decode(ids)
print(text)

```

----------

### Save / Load
Training automatically saves cache files.

Later runs can directly load:

```python
tokenizer.encoder(...)
tokenizer.decode(...)

```

**without retraining.**

----------

# Limitations

This repository intentionally keeps things simple.

It does **not** aim to include:

-   Rust/C++ speed optimizations
    
-   Parallel training
    
-   Streaming tokenization
    
-   Full GPT tokenizer parity
    
    
That is by design. The focus is understanding the algorithm clearly.

----------

# Future Improvements

Possible upgrades:

-   CLI interface

-   Compare against `tiktoken`
      
-   Faster pair-count updates
        
----------

# Final Verdict

Mini-BPE-Algorithm is a strong educational implementation of a real NLP primitive.

It demonstrates that you do not need to treat tokenizers as magic—you can build one, inspect it, and understand it.

That knowledge transfers directly to modern LLM systems.

----------

# References

1. [OpenAI GPT-2](https://github.com/openai/gpt-2)

2.  [OpenAI GPT-3 Encoder](https://github.com/openai/GPT-3-Encoder)
    
3.  [Karpathy Minbpe Repository](https://github.com/karpathy/minbpe)

4.  [Andrej Karpathy's YT Video on Tokenizers](https://youtu.be/zduSFxRajkE?si=sCkEEwSAk1LIR7o2) 
   
---
# Author
**Bhavesh Gaikwad**
LinkedIn - [Click Here](https://www.linkedin.com/in/bhavesh-g-663a34257/)
GitHub - [Click Here](https://github.com/BhaveshG29)
