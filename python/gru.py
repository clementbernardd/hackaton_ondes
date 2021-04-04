import torch 
import torch.nn as nn 
import torch.optim as optim
import os 

class GRU(nn.Module) : 
  def __init__(self, input_dim,embed_dim , hidden_dim, output_size,name = 'gru') : 
    super(GRU,self).__init__()
    self.checkpoint = name
    self.embed_dim = embed_dim
    self.embed = nn.Linear(input_dim, embed_dim)
    self.gru = nn.GRU(input_size = embed_dim , hidden_size = hidden_dim,num_layers=1, batch_first = True )
    self.out = nn.Linear(hidden_dim, output_size)

  def forward(self, x) :
    x = self.embed(x.float())
    x = x.view(-1, 1,self.embed_dim )
    gru_output , h_n = self.gru(x)
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
