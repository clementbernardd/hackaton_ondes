import torch
import torch.nn as nn
import torch.optim as optim
import os


class MLP_STACK(nn.Module) :
  def __init__(self, input_dim, hidden_dim, output_size, name = 'mlp') :
    super(MLP_STACK,self).__init__()
    self.checkpoint = name
    self.input_dim = input_dim
    self.fc1 = nn.Linear(17 * input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.out = nn.Linear(hidden_dim, output_size)

  def forward(self, x) :
    x = x.view(-1, 17 * self.input_dim)
    x = nn.ReLU()(self.fc1(x.float()))
    x = nn.ReLU()(self.fc2(x))
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
