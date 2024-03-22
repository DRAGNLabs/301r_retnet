from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt-2")

text = "Hello, my dog is cute"

encoded_input = tokenizer(text, return_tensors="pt")

print(encoded_input)