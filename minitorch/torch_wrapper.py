from collections import Counter
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from IPython.display import HTML
from operator import itemgetter
torch.manual_seed(42)


class TorchWrapper:
  def __init__(self, input_data, 
               output_data,
               custom_dataset_config,
               transforms=None,
               val_size=0.15, test_size=0.5,
               batch_sizes=(32,32,1), num_workers=(0,0,0),
               preprocess_data=False):
    try:
      assert 1-val_size-test_size > 0
    except:
      raise ValueError(f"'val_size' ({val_size}) + 'test_size' ({test_size}) is more than 1.")
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.input_data = input_data
    self.output_data = output_data
    self.custom_dataset_config = custom_dataset_config
    if transforms:
      try:
        assert isinstance(transforms, dict)
      except:
        raise TypeError("Passed Transforms must either be None or a dictionary.")
    self.transforms = transforms
    self.val_size = val_size
    self.test_size = test_size/(1-val_size)
    self.batch_sizes = batch_sizes
    self.num_workers = num_workers
    if preprocess_data:
      if isinstance(self.output_data[0], str):
        self.target2ind = dict(zip(set(output_data), range(len(set(output_data)))))
        self.ind2target = {v:k for k,v in self.target2ind.items()}
        self.output_data = np.array([self._one_hot_encode(label) for label in self.output_data])
        self.ohe = True
        self.stratify = True
        self.allow_target_weights = True
      else:
        self.target2ind = {}
        self.ind2target = {}
        self.ohe = False
        self.stratify = False
        self.allow_target_weights = False
      self._split_data()
      self._load_data()    

  def load_checkpoint(self, net,
                      checkpoints_path, model_name, load_type='eval'):
    checkpoint = torch.load(checkpoints_path+model_name)
    model_checkpoint = torch.load(checkpoints_path+'base_model.pt')
    self.net = net
    if model_checkpoint.get('target2ind') is not None:
      self.target2ind = model_checkpoint['target2ind']
      self.ind2target = {v:k for k,v in self.target2ind.items()}
      self.output_data = np.array([self._one_hot_encode(label) for label in self.output_data])
      self.ohe = True
      self.stratify = True
      self.allow_target_weights = True
    else:
      self.ohe = False
      self.stratify = False
      self.allow_target_weights = False
    self.optimizer = model_checkpoint['optimizer']
    self.net.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.net.to(self.device)
    if load_type == 'eval':
      for parameter in self.net.parameters():
        parameter.requires_grad = False
      self.net.eval()
    elif load_type == 'train':
      self.net.train()
    else:
      raise ValueError("Please select either \'eval\' or \'train\' for load_type.")
    print('Loaded {} in {} mode.'.format(model_name, load_type))
    train_val_curve = base64.b64encode(open(checkpoints_path+'train_val_curve.png', 'rb').read()).decode('utf-8')
    display(HTML(f"""<img src="data:image/png;base64,{train_val_curve}">"""))
    self._split_data()
    self._load_data()

  def _one_hot_encode(self, input_data):
    one_hot_encoding = np.zeros(len(self.target2ind))
    one_hot_encoding[self.target2ind[input_data]] = 1
    return one_hot_encoding

  def _split_data(self):
    """Split into stratified training/validation/testing sets.
    """
    inds = np.array(range(len(self.input_data)))

    train_inds, self.val_inds  = train_test_split(inds, 
                                                  stratify=self.output_data if self.stratify else None,
                                                  test_size=self.val_size, 
                                                  random_state=42)

    self.train_inds, self.test_inds  = train_test_split(train_inds, 
                                                        stratify=self.output_data[train_inds] if self.stratify else None, 
                                                        test_size=self.test_size, 
                                                        random_state=42)
    
    self.X_train = self.input_data[self.train_inds]
    self.y_train = self.output_data[self.train_inds]

    self.X_val = self.input_data[self.val_inds]
    self.y_val = self.output_data[self.val_inds]

    self.X_test = self.input_data[self.test_inds]
    self.y_test = self.output_data[self.test_inds]

    print("Data is split:")
    print(f"Training shape: {self.X_train.shape, self.y_train.shape}")
    print(f"Validation shape: {self.X_val.shape, self.y_val.shape}")
    print(f"Testing shape: {self.X_test.shape, self.y_test.shape}")

  def _load_data(self):
    """Load data into DataSets and DataLoaders.
    """
    self.trainset = self.custom_dataset_config(self.X_train, self.y_train, self.device, 
                                                  transform=self.transforms.get('train') if self.transforms else None)
    self.valset = self.custom_dataset_config(self.X_val, self.y_val, self.device, 
                                                  transform=self.transforms.get('validation') if self.transforms else None)
    self.testset = self.custom_dataset_config(self.X_test, self.y_test, self.device, 
                                                  transform=self.transforms.get('test') if self.transforms else None)

    self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_sizes[0] if self.batch_sizes[0] != -1 else len(self.trainset),
                                                   shuffle=True, num_workers=self.num_workers[0])
    self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_sizes[1] if self.batch_sizes[1] != -1 else len(self.valset),
                                                 shuffle=True, num_workers=self.num_workers[1])
    self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_sizes[2] if self.batch_sizes[2] != -1 else len(self.testset),
                                                  shuffle=True, num_workers=self.num_workers[2])
    try:
      _ = [(inputs, labels) for inputs, labels in self.custom_dataset_config([self.trainset.x[0]], 
                                                                            [self.trainset.y[0]], 
                                                                            self.device, 
                                                                            self.transforms.get('train') if self.transforms else None)]
    except:
      raise ValueError("Something went wrong in the DataLoaders.")
    
    print("\nData is loaded into DataLoaders.")

  def load_net(self, net, 
               chosen_criterion, 
               chosen_optimizer, chosen_optimizer_params,
               optimizer_network_parameters=None,
               weights=None):
    """Load the chosen network.
    """
    self.weights = weights
    self.base_net = deepcopy(net)
    self.net = net
    self.net.to(self.device)
    if optimizer_network_parameters:
      params = optimizer_network_parameters
    else:
      params = self.net.parameters()
    self.optimizer = chosen_optimizer(params=params, **chosen_optimizer_params)
    try:
      with torch.no_grad():
        for inputs, labels in self.custom_dataset_config([self.trainset.x[0]], 
                                                         [self.trainset.y[0]], 
                                                         self.device, 
                                                         self.transforms['train'] if self.transforms else None):
          _ = self.net(inputs[None, :])
    except:
      raise ValueError("Test forward pass failed, check network architecture.")

    if self.allow_target_weights:
      if self.weights == 'inverted':
        weights_arr = np.zeros(self.y_train[0].shape[0]) 
        for target_arr in self.y_train:
          target_ind = np.argmax(target_arr)
          weights_arr[target_ind] += 1
        weights_arr = 1/weights_arr
        chosen_weights = torch.tensor(weights_arr, 
                                      device=self.device, dtype=torch.float)
      elif self.weights == 'equal':
        chosen_weights = torch.tensor(np.ones(self.y_train[0].shape[0]), 
                                      device=self.device, dtype=torch.float)
      else:
        raise ValueError("Please select either \'equal\' or \'inverted\' weights.")
      self.criterion = chosen_criterion(weight=chosen_weights)

    print("\nNetwork is loaded.\n")

  def train(self, epochs, log_mini_batches=2000,
            models_dir='/content/model_checkpoints'):
    self.net.train()
    model_path = models_dir+datetime.utcnow().strftime('/%Y%m%dT%H%M%S_models')
    if not os.path.exists(models_dir):
      os.mkdir(models_dir)
    if not os.path.exists(model_path):
      os.mkdir(model_path)
    torch.save({'optimizer': self.optimizer,
                'target2ind':self.target2ind},
               model_path+'/base_model.pt')
    self.epochs = epochs
    self.training_loss = []
    self.num_minibatches = []
    self.validation_loss = []
    running_loss = 0.0
    for epoch in range(1, epochs+1):
      for i, data in enumerate(self.trainloader, 0):

        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels.flatten())
        loss.backward()
        self.optimizer.step()

        # get statistics every log_mini_batches
        running_loss += loss.item()
        if i % log_mini_batches == (log_mini_batches-1):
          
          # log the running training loss
          self.training_loss.append(running_loss / log_mini_batches)
          
          # log the running validation loss
          self.net.eval()
          with torch.no_grad():
            running_val_loss = 0.0
            for i_val, data_val in enumerate(self.valloader, 0):
              inputs_val, labels_val = data_val
              outputs_val = self.net(inputs_val)
              loss_val = self.criterion(outputs_val, labels_val.flatten()).item()
              running_val_loss += loss_val
          self.net.train()
          self.validation_loss.append(running_val_loss / len(self.valloader))

          self.num_minibatches.append((epoch-1) * len(self.trainloader) + i)
          
          print('[%d, %5d] train_loss: %.3f | val_loss: %.3f' %
                  (epoch, i + 1, running_loss / log_mini_batches, running_val_loss / len(self.valloader)))

          # save training curve
          self.plot_training_curves(show=False, save=model_path+'/train_val_curve.png')
            
          running_loss = 0.0
        
      # save a checkpoint each epoch
      torch.save({
                  'epoch': epoch,
                  'model_state_dict': self.net.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict()
                  }, model_path+'/epoch{}_model.pt'.format(epoch))

    self.plot_training_curves()
    print('Finished Training')

  def plot_training_curves(self, show=True, save=None):
    fig = plt.figure()
    plt.plot(self.num_minibatches, self.training_loss, label='Training')
    plt.plot(self.num_minibatches, self.validation_loss, label='Validation')
    plt.legend()
    plt.title('Training and Validation Curves')
    plt.xlabel('Mini-Batches')
    plt.ylabel('Loss')
    if save:
      plt.savefig(save)
    if show:
      plt.show();
    else:
      plt.close()

  def predict(self, inputs, softmax=False, 
              interence_transform=False):
    # TODO: implement test-time augmentation. Just set transform probabilities below 1?
    self.net.eval()
    with torch.no_grad():
      if interence_transform:
        if (self.transforms is not None) and (self.transforms.get('inference') is not None):
          inputs = self.transforms.get('inference')(image=np.array(inputs))['image']
      if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs)
      outputs = self.net(inputs.float().to(self.device)).cpu().detach()
      if softmax:
        return F.softmax(outputs[0], dim=0).numpy()
      return outputs.numpy()

  def evaluate(self):
    """Calculate accuracy on the train, validation, and testing sets.
    """
    # TODO: This will only work for classification problems
    self.net.eval()
    self.micro_accuracy_dict = {t:{'count':0, 'correct':0} for t in self.target2ind}
    self.micro_hitn_dict = {t:{'count':0, 'hit':0} for t in self.target2ind}
    self.test_set_confusion_matrix = np.zeros((len(self.target2ind), len(self.target2ind)))
    y_test_value_counts = Counter([np.argmax(t) for t in self.y_test])
    for data_group in [['Train', self.trainloader],
                       ['Validation', self.valloader],
                       ['Test', self.testloader]]:
      dataset_name, data = data_group
      print(f'Evaluating {dataset_name} set...')
      correct = 0
      hit = 0
      count = 0
      for input_data, targets in tqdm(data, position=0, leave=True):
        outputs = self.predict(input_data)
        for output, target in zip(outputs, targets):
          sorted_output = np.argsort(output).flatten()[::-1]
          predicted_ind = sorted_output[0]
          target_ind = target.item()
          hit_at_n = np.where(sorted_output == target_ind)[0][0]
          if dataset_name == 'Test':
            self.test_set_confusion_matrix[target_ind][predicted_ind] += 1
            self.micro_hitn_dict[self.ind2target[target_ind]]['hit'] += hit_at_n
            self.micro_hitn_dict[self.ind2target[target_ind]]['count'] += 1
          if target_ind == predicted_ind:
            correct += 1
            if dataset_name == 'Test':
              self.micro_accuracy_dict[self.ind2target[target_ind]]['correct'] += 1
          if dataset_name == 'Test':
            self.micro_accuracy_dict[self.ind2target[target_ind]]['count'] += 1
          count += 1
          if count > 1:
            hit *= (count-1)
            hit += hit_at_n
            hit /= count
          else:
            hit = hit_at_n
      print(f"\nAverage {dataset_name} Accuracy: {100 * correct / count:.2f}%")
      print(f"Average {dataset_name} Hit@N: {hit:.1f}\n")

    for count_dict in self.micro_accuracy_dict.values():
      if count_dict['count'] > 0:
        count_dict['accuracy'] = 100 * count_dict['correct'] / count_dict['count']

    for count_dict in self.micro_hitn_dict.values():
      if count_dict['count'] > 0:
        count_dict['hit'] /= count_dict['count']

    self.test_set_confusion_matrix_normed = deepcopy(self.test_set_confusion_matrix)
    for n, row in enumerate(self.test_set_confusion_matrix_normed):
      row /= y_test_value_counts[n]

    figure = plt.figure(figsize=(17,15))
    sns.heatmap(self.test_set_confusion_matrix_normed, cmap='rocket_r')
    plt.title('Micro-Accuracy Test Set Predictions Heatmap')
    ticks = [self.ind2target[i] for i in range(len(self.test_set_confusion_matrix_normed))]
    plt.xticks([r+0.5 for r in range(len(self.test_set_confusion_matrix_normed))], ticks, rotation=90, size=8)
    plt.yticks([r+0.5 for r in range(len(self.test_set_confusion_matrix_normed))], ticks, rotation=0, size=8)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    plt.show();

    sorted_hit = sorted([[card, hitn_dict['hit']] for card, hitn_dict in self.micro_hitn_dict.items()], key=itemgetter(1))
    max_hit_lim = int(np.ceil(max([hitn_dict['hit'] for hitn_dict in self.micro_hitn_dict.values()]))+1)
    figure = plt.figure(figsize=(max(max_hit_lim, 5), 7))
    plt.bar(range(len(sorted_hit)), [sp[1] for sp in sorted_hit])
    plt.xticks(range(len(sorted_hit)), [sp[0] for sp in sorted_hit], rotation=90)
    for n, group in enumerate(sorted_hit):
      card, hit = group
      plt.text(n, hit, f"{hit:.0f}", ha='center', va='bottom')
    plt.title(f'Average Test Set Hit @ N by Target Class\nMax is {len(sorted_hit)}')
    plt.xlabel('Target Class')
    plt.ylabel('Hit @ N')
    plt.show();