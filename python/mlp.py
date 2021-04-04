import torch 
import torch.nn as nn
import torch.optim as optim 
import os 

class MLP(nn.Module) : 
  def __init__(self, input_dim, hidden_dim, output_size,name = 'mlp') : 
    super(MLP,self).__init__()
    self.checkpoint = name
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.out = nn.Linear(hidden_dim, output_size)

  def forward(self, x) :
    x = nn.ReLU()(self.fc1(x.float()))
    x = self.out(x)
    return x

  def save_checkpoint(self) :
      print('--- Save model checkpoint ---')
      torch.save(self.state_dict(), self.checkpoint)

  def load_checkpoint(self, gpu = True ) :

      print('--- Loading model checkpoint ---')
      if torch.cuda.is_available() and gpu :
          self.load_state_dict(torch.load(self.checkpoint))
      else :
          self.load_state_dict(torch.load(self.checkpoint,map_location=torch.device('cpu')))
