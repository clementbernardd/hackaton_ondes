import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_data(data, labels , p = 0.8 , shuffle = True ) :
  ''' Split data and labels into training and validation '''
  if shuffle :
    indexes_random = np.random.permutation(len(data))
    data, labels = data[indexes_random], labels[indexes_random]
  index_validation = int(p * len(data))
  train_data, validation_data = data[:index_validation], data[index_validation:]
  train_labels , validation_labels = labels[:index_validation], labels[index_validation:]

  return train_data, train_labels, validation_data, validation_labels


def stack_trames(data , N = 100) :
    ''' Convert the trames into block of 100 '''
    n_trames = int(data.shape[0]/N)
    new_data  = []
    for i in range(n_trames - 1) :
        new_data.append(data[N * i : N * (i+1)])
    return new_data


def stack_trames_labels(data : np.array, labels : np.array , N = 100) -> (np.array, np.array) :
    ''' Convert the data into block of 100 '''
    letters = np.arange(42)
    new_data = []
    new_labels = []

    for letter in letters :
        data_letter = data[np.where(labels == letter)[0]]
        new_letter = stack_trames(data_letter, N)

        new_data.extend(new_letter)
        new_labels.extend([letter for i in range(len(new_letter))])

    return np.array(new_data), np.array(new_labels)




def create_noise_frames(trame_a, trame_b, N = 25) :
  ''' Create noise from the trame b into the trame a
      N is the number of frames from b that goes into trame a
  '''
  c = np.random.uniform()
  if  c < 0.25 :
    return (trame_a + trame_b)/2
  elif c < 0.5 :
    return (trame_a - trame_b)/2
  elif c < 0.75 :
    return (trame_a * trame_b)
  else :
    return (trame_a + trame_b)




def add_noise(data : np.array, labels : np.array, coeff : int , n_frames : int ,p = 0.25 )  -> (np.array, np.array) :
  ''' Add noise in the trames '''
  new_data = list(data)
  new_labels = list(labels)
  augmentation = int(coeff * len(new_data))
  # Number of frames with noise
  N = int(p * n_frames)
  for letter in np.arange(42) :
    # Take c times the trame from a label and add noise with other caracters
    data_letter = data[np.where(labels == letter)[0]]
    for c in range(coeff) :
      # ADD NOISE
      data_c = data[np.where(labels == letter)[0]]
      index_c = np.random.choice(np.arange(len(data_c)))
      current_data = data[index_c]
      # Get trames from other caracters
      letter_new = np.random.choice([i for i in range(42) if i!=letter])
      trames_new = data[np.where(labels == letter_new)[0]]
      index_new = np.random.choice(np.arange(len(trames_new)))
      trame_to_add = trames_new[index_new]

      new_trame = create_noise_frames(current_data, trame_to_add, N = N)
      new_data.append(new_trame)
      new_labels.append(letter)
  return np.array(new_data), np.array(new_labels)


def get_prediction_all_dataset(model, data_loader, device ) :
    ''' Predict the model over all the data loader '''
    predictions = []
    y_true = []
    model = model.to(device)

    for x,y in data_loader :
        with torch.no_grad() :
            x,y = x.to(device), y.to(device)
            y_pred = model.forward(x)
            predictions.extend(torch.argmax(y_pred, axis = 1).detach().cpu().numpy())
            y_true.extend(y.detach().cpu().numpy())
    return np.array(predictions) , y_true


def plot_mulitple_confusion_matrix(list_confusion, names, nrows=1,ncols=3) :
  ''' Plot multiple confusion matrix '''
  figure, ax = plt.subplots(figsize = (20,12) , nrows=nrows,ncols=ncols)
  ax = np.array(ax)

  for i,conf in enumerate(list_confusion) :

      im = ax.flatten()[i].imshow(conf, cmap = 'magma')
      cbar = ax.flatten()[i].figure.colorbar(im, ax=ax.flatten()[i])
      ax.flatten()[i].set_title(names[i])
      ax.flatten()[i].grid(False)
      ax.flatten()[i].set_xlabel('Y predicted')
      ax.flatten()[i].set_ylabel('Y true')

  plt.tight_layout()
  plt.show()



def plot_chart_bart_score(scores_all, name = 'F1 score') :
  ''' Plot char bart of the score of the different models '''
  # Get the data
  score = {}
  for method in scores_all :
    score_method = [x for x in list(scores_all[method].values())]
    score[method] = score_method

  # Plot the data
  width = 0
  width_ = 0.2
  plt.subplots(figsize = (14,8))
  X = np.arange(3)

  for model in score :
    plt.bar(X+width, score[model], width = width_, label = model)
    width+=width_

  plt.ylim([0,1])
  plt.xticks(X,['Digits', 'Not Digits', 'All'])
  plt.ylabel(name)
  plt.grid(True)
  plt.title(name+' for the different models')
  plt.legend()





def get_accuracy_digits(y_pred, y, metric = f1_score) :
  ''' Return the accuracy for the digits and without the digitis '''
  y = np.array(y)
  index_digits = []
  index_not_digits = []

  for i in range(len(y)) :
    if i >= 1 and i < 11 :
      index_digits.extend(np.where(y == i)[0])
    else :
      index_not_digits.extend(np.where(y == i)[0])

  y_digits = np.array(y)[index_digits]
  y_not_digits = np.array(y)[index_not_digits]

  y_pred_digits = np.array(y_pred)[index_digits]
  y_pred_not_digits = np.array(y_pred)[index_not_digits]


  if metric.__name__ == 'f1_score' :
    acc_digits = metric(y_digits, y_pred_digits, average = 'micro')
    acc_not_digits = metric(y_not_digits, y_pred_not_digits, average = 'micro')
    acc_all = metric(y, y_pred, average = 'micro')
  else :
    acc_digits = metric(y_digits, y_pred_digits)
    acc_not_digits = metric(y_not_digits, y_pred_not_digits)
    acc_all = metric(y, y_pred)

  name = metric.__name__

  scores = {name+'_digits' : acc_digits, name+'_not_digits': acc_not_digits, name+'_all' : acc_all}

  return scores



def plot_chart_bart_score(scores_all, name = 'F1 score', width_ = 0.2) :
    ''' Plot char bart of the score of the different models '''
    # Get the data
    score = {}
    for method in scores_all :
        score_method = [x for x in list(scores_all[method].values())]
        score[method] = score_method
    # Plot the data
    width = 0
    plt.subplots(figsize = (14,8))
    X = np.arange(3)

    for model in score :
        plt.bar(X+width, score[model], width = width_, label = model)
        width+=width_

    plt.ylim([0,1])
    plt.xticks(X,['Digits', 'Not Digits', 'All'])
    plt.ylabel(name)
    plt.grid(True)
    plt.title(name+' for the different models')
    plt.legend()


def plot_multiple_predictions(y_preds,names , nrows = 1 , ncols = 3) :
  ''' Plot multiple predictions '''
  figure, ax = plt.subplots(figsize = (22,8) , nrows=nrows,ncols=ncols)
  ax = np.array(ax)

  for i,y in enumerate(y_preds) :
      ax.flatten()[i].scatter(np.arange(len(y)), y)
      ax.flatten()[i].set_title(names[i])
      ax.flatten()[i].grid(True)
      ax.flatten()[i].set_xlabel('Trame')
      ax.flatten()[i].set_ylabel('Y predicted')

  plt.tight_layout()
  plt.show()

def plot_proba(y_test_dict, nrows = 2, ncols = 2) :
  ''' Plot the different probabilities for different models '''
  figure, ax = plt.subplots(figsize = (24,14) , nrows=nrows,ncols=ncols)
  ax = np.array(ax)

  for i,model in enumerate(y_test_dict) :

      y = y_test_dict[model]
      im = ax.flatten()[i].imshow(np.flip(np.rot90(y),0), aspect = 'auto', interpolation='none', origin='lower', cmap='viridis')
      ax.flatten()[i].locator_params(axis="x", nbins=10)
      ax.flatten()[i].locator_params(axis="y", nbins=42)
      ax.flatten()[i].set_xlabel('Trame')
      ax.flatten()[i].set_ylabel('Class')
      ax.flatten()[i].set_title(model)

  plt.show()





def plot_multiple_prediction(y_dict, nrows = 1, ncols =2, name = 'GRU') :
  ''' Plot multiple predictions '''
  figure, ax = plt.subplots(figsize = (16,10) , nrows=nrows,ncols=ncols)
  ax = np.array(ax)
  y_pred = list(y_dict.values())
  names = list(y_dict.keys())

  for i,y in enumerate(y_pred) :

      ax.flatten()[i].grid(True)
      ax.flatten()[i].scatter( np.arange(len(y)),y , marker = '+', label = 'Y predicted')
      ax.flatten()[i].set_title('Y pred with {} with {} frames'.format(name, names[i]))
      ax.flatten()[i].legend()
      ax.flatten()[i].set_xlabel('Trame')
      ax.flatten()[i].set_ylabel('Class')

  plt.title('Prediction with {}'.format(name))
  plt.tight_layout()
  plt.show()




def plot_proba(y_test_dict, nrows = 3, ncols = 2, name = 'GRU') :
  ''' Plot the different probabilities for different models '''
  figure, ax = plt.subplots(figsize = (24,14) , nrows=nrows,ncols=ncols)
  ax = np.array(ax)

  for i,frame in enumerate(y_test_dict) :

      y = y_test_dict[frame]
      im = ax.flatten()[i].imshow(np.flip(np.rot90(y),0), aspect = 'auto', interpolation='none', origin='lower', cmap='viridis')
      ax.flatten()[i].locator_params(axis="x", nbins=10)
      ax.flatten()[i].locator_params(axis="y", nbins=42)
      ax.flatten()[i].set_xlabel('Trame')
      ax.flatten()[i].set_ylabel('Class')
      ax.flatten()[i].set_title('Frame {} for {}'.format(frame, name))

  plt.show()


# Distance between probas of differents trames in the window
def dist_window(x) :
    dist = 0
    count = 0
    for index,vect in enumerate(x):
        for index_,vect_ in enumerate(x):
            if index_ != index:
                dist+= np.linalg.norm(vect-vect_)
                count+=1
    if (count>0):
        return dist/count
    else:
        return 0

# Compute this distance with a sliding window
def moving_average_dist(x,window_size = 10):
    dists=[]
    for i in range(x.shape[0]) :
        end=min(x.shape[0], i + window_size)
        dists.append(dist_window(x[i:end]))
    return np.array(dists)

# Smooth the probas with a moving average
def moving_average(x,window_size = 10):
    means=[]
    for i in range(x.shape[0]) :
        start = max(0 , i - window_size )
        means.append(np.mean(x[start:i+1], axis=0))
    return np.array(means)


def find_letters_range(loginmdp_predict_proba,window=30):
    mav = moving_average(loginmdp_predict_proba,window)
    mmd = moving_average_dist(mav,window)
    peaks, _ = find_peaks(mmd, height=0.15, distance=10)
    plt.figure(figsize = (15,5))
    plt.plot(mmd)
    plt.plot(peaks, mmd[peaks], "x")
    plt.show()

    auto_bins = []
    former_value=None
    for value in peaks:
        if former_value is not None:
            auto_bins.append([former_value,value])
        former_value=value

    return mav, mmd, auto_bins, peaks








def plot_moving_average(data, start=0, end = None, nbins_=50, shapeW=30, shapeH=10) :
    if end==None:
        END = data.shape[0]
    else:
        END = end
    START = start

    plt.figure(figsize = (shapeW,shapeH))
    im = plt.imshow(np.flip(np.rot90(data),0), aspect = 'auto', interpolation='none', origin='lower', cmap='viridis')
    ax = plt.gca()
    plt.locator_params(axis="x", nbins=nbins_)
    plt.locator_params(axis="y", nbins=42)
    plt.xlim(START,END)
    plt.show()


def plot_moving_average_dist(data_, start=0, end_=None, peaks=None):
    end = end_
    if end_==None:
        end = data_.shape[0]
    data = data_[start:end]

    figure, ax = plt.subplots(figsize = (20,5))
    ax.set_xlabel('Numéro de trame')
    ax.set_ylabel('Probabilité d\'appartenance à chaque catégorie')
    plt.plot(np.arange(0,len(data),1),data)
    plt.locator_params(axis="x", nbins=30)
    plt.locator_params(axis="y", nbins=10)
    plt.xlim(start,end)
    print(end)
    plt.show()




def get_indexes_trames(auto_bins, peaks_pred,peaks) :
  ''' Return the trames for lock, login and mdp '''
  indexes = {'lock' : [0], 'login' : [], 'mdp' : []}
  c = peaks_pred[0]
  i = 0
  for j , (peak, classe) in enumerate(zip(peaks, peaks_pred)) :
    if classe != c :
      indexes[  list(indexes.keys())[i] ].append(j)
      if i < 3 :
        indexes[  list(indexes.keys())[i+1] ].append(j)
      c = classe
      i+=1
  indexes['mdp'].append(len(auto_bins)-1)
  return indexes




def get_distance_bins(peaks) :
  ''' Return bins most probable '''
  auto_bins = []
  former_value=None
  for value in peaks:
      if former_value is not None:
          auto_bins.append([former_value,value])
      former_value=value
  return auto_bins




def plot_prediction_proba(self, y_pred_=None, start=0, end = None, nbins_=50, shapeW=30, shapeH=10) :
    if y_pred_ is None:
        y_pred=self.loginmdp_predict_probas
    else:
        y_pred=y_pred_
    if end==None:
        END = y_pred.shape[0]
    else:
        END = end
    START = start

    plt.figure(figsize = (shapeW,shapeH))
    im = plt.imshow(np.flip(np.rot90(y_pred),0), aspect = 'auto', interpolation='none', origin='lower', cmap='viridis')
    ax = plt.gca()
    plt.locator_params(axis="x", nbins=nbins_)
    plt.locator_params(axis="y", nbins=y_pred.shape[1])
    plt.xlim(START,END)
    plt.show()





def plot_probability_words(mdp) :
  grid = np.flip(np.rot90([mdp[i][1] for i in range(len(mdp)-1, -1, -1)]))

  fig, ax= plt.subplots(figsize=(20,3))
  plt.imshow(grid, interpolation ='none', aspect = 'auto')
  for (j, i), _ in np.ndenumerate(grid):
      label = mdp[i][0][j][:3]
      if len(label)>1:
          ax.text(i,j,label,ha='center',va='center', c='white',fontweight='bold', fontsize=8)
      else:
          ax.text(i,j,label,ha='center',va='center', c='white',fontweight='bold')
      ax.set_xlabel('Touches pressées', fontsize=13)
      ax.set_ylabel('Ordre décroissant des 5\n touches les plus probables', fontsize=13)
  plt.colorbar()
  plt.show()
