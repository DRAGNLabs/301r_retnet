import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2
import torch.nn.functional as F

class Tokenizer():
    def __init__(self, vocab, tokenizer, tokens_to_text):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.tokens_to_text = tokens_to_text
    
    def stoi(self, text):
        return [self.vocab[token] for token in self.tokenizer(text)]
    
    def itos(self, token_indices):
        return self.tokens_to_text(token_indices)

def load_wikitext2(max_seq_len, batch_size):
    """ Loads the WikiText2 dataset and returns the train, validation and test data loaders
    Args:
        max_seq_len (int): Maximum sequence length
        batch_size (int): Batch size
    Returns:
        train_loader (DataLoader): Training data loader
        valid_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
    """
    # Tokenizer function
    tokenizer = get_tokenizer('basic_english')

    # Function to yield tokens from dataset
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    # Load the dataset
    train_iter, valid_iter, test_iter = WikiText2()

    # Build vocabulary from training set
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    # Create a reverse mapping from indices to tokens
    index_to_token = {index: token for token, index in vocab.get_stoi().items()}


    def tokens_to_text(token_indices):
        return ' '.join([index_to_token[index] for index in token_indices])
        
    tokenizer = Tokenizer(vocab, tokenizer, tokens_to_text)

    # Function to process each article
    def data_process(raw_text_iter):
        processed_data = []
        for text in raw_text_iter:
            # Tokenize and numericalize
            numericalized_text = tokenizer.stoi(text)
            # Pad and possibly truncate the sequence
            padded = F.pad(torch.tensor(numericalized_text, dtype=torch.long),
                        (0, max_seq_len - len(numericalized_text)),
                        value=vocab["<pad>"])
            if len(padded) > max_seq_len:
                padded = padded[:max_seq_len]
            processed_data.append(padded)
        return processed_data

    # Process the datasets
    train_data = data_process(train_iter)
    valid_data = data_process(valid_iter)
    test_data = data_process(test_iter)

    # Custom Dataset class
    class WikiTextDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            # Ensure the sequence is of MAX_SEQ_LEN
            item_padded = F.pad(item, (0, max_seq_len - len(item)), value=vocab["<pad>"])
            # Input is the entire sequence
            input = item_padded
            # Target is the same sequence shifted by one position and padded
            target = F.pad(item_padded[1:], (0, 1), value=vocab["<pad>"])
            return input, target

    # Create datasets
    train_dataset = WikiTextDataset(train_data)
    valid_dataset = WikiTextDataset(valid_data)
    test_dataset = WikiTextDataset(test_data)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, tokenizer
