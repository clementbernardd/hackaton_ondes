

import torch
import torch.nn as nn
import torch.optim as optim
import os

class GRU_NOISE(nn.Module) :
  def __init__(self, input_dim,embed_dim , hidden_dim, output_size, name = 'gru_noise') :
    super(GRU_NOISE,self).__init__()
    self.checkpoint = os.path.join(os.getcwd(),name)
    self.embed_dim = embed_dim
    self.input_dim = input_dim
    self.gru = nn.GRU(input_size = input_dim , hidden_size = hidden_dim,num_layers=2, batch_first = True )
    self.out = nn.Linear(hidden_dim, output_size)

  def forward(self, x) :
    x = x.view(-1,  17 , self.input_dim)
    gru_output , h_n = self.gru(x.float())
    out = self.out(gru_output)[:,-1,:]
    return out

  def save_checkpoint(self) :
      print('--- Save model checkpoint ---')
      torch.save(self.state_dict(), self.checkpoint)

  def load_checkpoint(self, gpu = True ) :

      print('--- Loading model checkpoint ---')
      if torch.cuda.is_available() and gpu :
          self.load_state_dict(torch.load(self.checkpoint))
      else :
          self.load_state_dict(torch.load(self.checkpoint,map_location=torch.device('cpu')))
