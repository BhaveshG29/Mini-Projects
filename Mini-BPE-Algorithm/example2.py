from tokenizer import BPE

tokenizer = BPE()

tests = [
    "hello",
    "HELLO",
    "Hello world!",
    "नमस्ते दुनिया",
    "Python 🐍",
    "12345",
    "Mixed हिंदी English 😀"
]

print("Encode==Decode?\t | Length of Token List | Decoded String")
for t in tests:
    ids = tokenizer.encoder(t)
    out = tokenizer.decode(ids)
    print(f"{t == out}\t\t | {len(ids)}\t\t\t| {t}")
