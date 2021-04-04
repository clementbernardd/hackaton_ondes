import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Pics_Dataset(Dataset):
  def __init__(self, data, labels):
      super(Dataset, self).__init__()
      self.data = data 
      self.labels = labels 
        
  def __len__(self):
      return len(self.data)
    
  def __getitem__(self, index):
    return torch.from_numpy(self.data[index]),torch.from_numpy(np.array(self.labels[index]))
