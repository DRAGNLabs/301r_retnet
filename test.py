from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Hello, my dog is cute"

encoded_input = tokenizer(text, return_tensors="pt")

print(encoded_input)

random_tokens = [1, 4, 5, 6, 7, 8, 9, 10]
random_tokens