import torch
from torch.utils.data import Dataset, DataLoader

from utils.enwik8_utils import enwik8

class Enwik8Chunks(Dataset):

    def __init__(self, split: str, block_size: int = 256):

        assert split in {"train", "val", "test"}, "invalid split name"

        # Context length
        self.block = block_size

        # Load the raw bytes, returns 3 tensors
        train_data, val_data, test_data = enwik8()
        
        # Select the appropriate split and convert to long tensor for CE
        if split == "train":
            self.data = train_data.long()
        elif split == "val":
            self.data = val_data.long()
        else:  # split == "test"
            self.data = test_data.long()

        # Length of the split
        self.L = len(self.data)

    def __len__(self):
        # Number of valid items in the dataset
        # A valid item is a block_size + 1 consecutive bytes (1 for the target)
        # All positions are valid until L - (block_size + 1) + 1
        # If the data is for some reason shorter than block size then return 0
        return max(0, self.L - self.block)

    def __getitem__(self, idx):
        # Ignore idx and do "random" items instead

        # Pick random start from valid options
        start_idx = torch.randint(0, self.L - self.block, (1,)).item()
        
        # Slice data
        chunk = self.data[start_idx:start_idx + self.block + 1]
        
        # Split into input / target (shift by one)
        x = chunk[:-1]  # all but last element
        y = chunk[1:]   # all but first element
  
        return {'input': x, 'target': y}


def get_loader(split: str,
               block_size: int,
               batch_size: int,
               num_workers: int = 2,
               pin_memory: bool = True):
    
    # Returns a DataLoader so the training is very mindful and very demure

    dataset = Enwik8Chunks(split, block_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),     # Doesn't matter, its always kinda shuffled anyway
        drop_last=True,                 # Discard the last incomplete batch to keep the same size
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
