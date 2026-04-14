from tokenizer import BPE

tokenizer = BPE(vocab_size=30000, n_merges=3000)

# Training on data folder
tokenizer.train("data", print_progress=True)

# This creates:
# vocab.json
# merges 

# Encode Custom Text
text = "Hello world! नमस्ते"

ids = tokenizer.encoder(text)

print("\n\nEncoded IDs:")
print(ids)

# Decoding Back The Custom Text
decoded_text = tokenizer.decode(ids)

print("\nDecoded Text:")
print(decoded_text)

# Verification of Does Encoded == Decoded Text
print("\nMatch:", text == decoded_text)
