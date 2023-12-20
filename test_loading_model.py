from train_model import RetNetModel
import torch
import torch.nn as nn
import utils
from datasets import load_wikitext2



'''
    --activation-dropout 0.0 \
    --checkpoint-activations False \
    --dropout 0.0 \
    --embed-dim 64 \
    --ffn-dim 64 \
    --fsdp True \
    --layers 2 \
    --lr 0.001 \
    --model retnet \
    --heads 8 \
    --seq-len 64 \
    --value-embed-dim 64 \
    --vocab-size 20480 \
    --device cuda \
    --epochs 1 \
    --batch-size 32 \
'''
args = {}
args["embed_dim"] = 64
args["value_embed_dim"] = 64
args["activation_dropout"] = 0.0
args["checkpoint_activations"] = False
args["dropout"] = 0.0
args["ffn_dim"] = 64
args["fsdp"] = True
args["layers"] = 2
args["lr"] = 0.001
args["model"] = "retnet"
args["heads"] = 8
args["seq_len"] = 64
args["device"] = "cuda"
args["epochs"] = 1
args["batch_size"] = 32
args["vocab_size"] = 20480
print("Instantiate RetNetModel base class.")
model = RetNetModel(embed_dim=args["embed_dim"],
                value_embed_dim=args["value_embed_dim"],
                retention_heads=args["heads"],
                ffn_dim=args["ffn_dim"],
                layers=args["layers"],
                dropout=args["dropout"],
                activation_dropout=args["activation_dropout"],
                vocab_size=args["vocab_size"],
                checkpoint_activations=args["checkpoint_activations"],
                fsdp=args["fsdp"],
                max_seq_len=args["seq_len"])
print("load state dictionary")
model.load_state_dict(torch.load("scripts/model_saved"))


 # Get DataLoaders
print("Get dataloaders.")
train_loader, valid_loader, test_loader, tokenizer = load_wikitext2(
            seq_len=args["seq_len"],
            batch_size=args["batch_size"],
            vocab_size=args["vocab_size"])
    
    #print("\nVocabulary:")
    #print(tokenizer.vocab.get_itos())



# Define the device to use
device = torch.device(args["device"])

start_string_list = [
            "<pad>",
            "One two three four five six seven eight nine ten "
            "One two three four five six seven eight nine ten "
            "One two three four five six seven eight nine ten "
            "One two three four five six seven eight nine ten "
            "One two three four five six seven eight nine ten "
            "One two three four five six seven eight nine ten "
            "One two three four five six seven eight nine ten "
            "One two three four five six seven eight nine ten "
            "One two three four five six seven eight nine ten "
            "One two three four five six seven eight nine ten ",
            "= = reception =",
            "= Valkyria Chronicles"]
print("Inference of text begins...")
generated_text = utils.generate_text(model, tokenizer, start_string_list, device, args["seq_len"], generation_length=100)
print("Text inference completed.")
for text in generated_text:
    print(text)

