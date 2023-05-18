
import torch
from torch.utils.data import Dataset



device = 'cuda'

class SimpleDataset(Dataset):
    def __init__(self, X, Y, L, H) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
        self.L = L
        self.H = H
 
    def __getitem__(self, index):
        index += self.H
        return dict(
            #dynamic = self.X[index : index+self.L+self.H], 
            dynamic = self.X[index-self.H : index+self.L], 
            lookback = self.Y[index : index+self.L],
            label = self.Y[index+self.L : index+self.L+self.H]
        )

    def __len__(self):
       #return len(self.X)-self.L-self.H+1
       return len(self.X)-self.L-self.H-self.H+1



def collate_fn(batch):
    return dict(
        dynamic = torch.stack([obj['dynamic'] for obj in batch], dim=0), 
        lookback = torch.stack([obj['lookback'] for obj in batch], dim=0),
        label = torch.stack([obj['label'] for obj in batch], dim=0)
    )