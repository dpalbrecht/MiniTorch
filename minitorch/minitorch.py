from collections import Counter
import itertools
import numpy as np
import sys
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

from . import metrics

class MiniTorch:
    def __init__(self, input_data,
                 output_data,
                 custom_dataset_config,
                 transforms=None,
                 val_size=0.15, test_size=0.5,
                 batch_sizes=(32,32,1), num_workers=(0,0,0),
                 **kwargs):
        """Instantiate MiniTorch.
        """
        # Validate validation/test splits are less than 100% of the dataset
        self.val_size = val_size
        self.test_size = test_size
        try:
            assert 1-self.val_size-self.test_size > 0
        except:
            raise ValueError(f"'val_size' ({val_size}) + 'test_size' ({test_size}) is more than 1.")
        self.test_size = self.test_size/(1-self.val_size)
        
        # Bind variables to the class
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
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        
        # Preprocess data if 'preprocess_data' parameter is not set to False
        # One-hot encode string dependent variables
        # Split into train/val/test sets
        # Load data into DataLoaders
        self.target2ind = None
        if kwargs.get('preprocess_data') is not False:
            if isinstance(self.output_data[0], str):
                self.target2ind = dict(zip(set(output_data), range(len(set(output_data)))))
                self.ind2target = {v:k for k,v in self.target2ind.items()}
                self.output_data = np.array([self._one_hot_encode(label) for label in self.output_data])
                self.stratify = True
                self.allow_target_weights = True
            self._split_data()
            self._load_data()

    @classmethod
    def load_checkpoint(cls,
                        net, checkpoints_path,
                        model_name, load_type='eval'):
        """Load a model checkpoint.
        """
        checkpoint = torch.load(checkpoints_path+model_name)
        model_checkpoint = torch.load(checkpoints_path+'base_model.pt')
        cls = cls(input_data=None, output_data=None,
                  custom_dataset_config=None,
                  **{'preprocess_data':False})
        cls.net = net
        if model_checkpoint.get('target2ind') is not None:
            cls.target2ind = model_checkpoint['target2ind']
            cls.ind2target = {v:k for k,v in cls.target2ind.items()}
            cls.allow_target_weights = True
        else:
            cls.allow_target_weights = False
        cls.optimizer = model_checkpoint['optimizer']
        cls.transforms = model_checkpoint['transforms']
        cls.criterion = model_checkpoint['criterion']
        cls.net.load_state_dict(checkpoint['model_state_dict'])
        cls.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cls.net.to(cls.device)
        if load_type == 'eval':
            for parameter in cls.net.parameters():
                parameter.requires_grad = False
            cls.net.eval()
        elif load_type == 'train':
            cls.net.train()
        else:
            raise ValueError("Please select either \'eval\' or \'train\' for load_type.")
        print('Loaded {} in {} mode.'.format(model_name, load_type))
        train_val_curve = base64.b64encode(open(checkpoints_path+'train_val_curve.png', 'rb').read()).decode('utf-8')
        display(HTML(f"""<img src="data:image/png;base64,{train_val_curve}">"""))
        cls.trainloader = model_checkpoint['trainloader']
        cls.valloader = model_checkpoint['valloader']
        cls.testloader = model_checkpoint['testloader']
        return cls

    def _one_hot_encode(self, input_data):
        """One-hot encode dependent variable if it's a string.
        """
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

        self.trainloader = torch.utils.data.DataLoader(self.trainset, 
                                                       batch_size=self.batch_sizes[0] if self.batch_sizes[0] != -1 else len(self.trainset),
                                                       shuffle=True, num_workers=self.num_workers[0])
        self.valloader = torch.utils.data.DataLoader(self.valset, 
                                                     batch_size=self.batch_sizes[1] if self.batch_sizes[1] != -1 else len(self.valset),
                                                     shuffle=True, num_workers=self.num_workers[1])
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.batch_sizes[2] if self.batch_sizes[2] != -1 else len(self.testset),
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
        
    def _save_model_artifacts(self, models_dir, model_path):
        """Save model artifacts before training begins.
        """
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save({'optimizer': self.optimizer,
                    'target2ind':self.target2ind,
                    'transforms':self.transforms,
                    'trainloader':self.trainloader,
                    'valloader':self.valloader,
                    'testloader':self.testloader,
                    'criterion':self.criterion},
                   model_path+'/base_model.pt')
        
    def _save_model_checkpoint(self, epoch, model_path):
        """Save a model checkpoint at each epoch.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path+'/epoch{}_model.pt'.format(epoch))

    def train(self, epochs, log_mini_batches=2000,
              models_dir='/content/model_checkpoints'):
        """Train the network.
        """
        self.net.train()
        model_path = models_dir+datetime.utcnow().strftime('/%Y%m%dT%H%M%S_models')
        self._save_model_artifacts(models_dir, model_path)
        self.epochs = epochs
        self.training_loss = []
        self.num_minibatches = []
        self.validation_loss = []
        running_loss = 0.0
        for epoch in range(1, epochs+1):
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels.flatten())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if i % log_mini_batches == (log_mini_batches-1):

                    # training loss
                    self.training_loss.append(running_loss / log_mini_batches)

                    # validation loss
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

            self._save_model_checkpoint(epoch, model_path)

        self.plot_training_curves()
        print('Finished Training')

    def plot_training_curves(self, show=True, save=None):
        """Plot training curves.
        """
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
        """Run an inference step.
        """
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

    def _instantiate_metrics(self, evaluation_metrics):
        """Create metrics dictionary for evaluation.
        """
        self.evaluation_dict = {}
        for evaluation_metric in evaluation_metrics:
            if metrics.metrics_lookup.get(evaluation_metric) is not None:
                self.evaluation_dict[evaluation_metric] = metrics.metrics_lookup.get(evaluation_metric)(self.evaluation_data)

    def _call_metrics(self, kwargs, predicted=None, truth=None, 
                      dataset_name=None, method='update'):
        """Call metrics methods: update, aggregate, report, and plot.
        """
        for metric_name, evaluation_class in self.evaluation_dict.items():
            if method == 'update':
                evaluation_class.update(predicted, truth, dataset_name)
            elif method == 'aggregate':
                evaluation_class.aggregate()
            elif method == 'report':
                evaluation_class.report(dataset_name)
            elif method == 'plot':
                evaluation_class.plot(kwargs.get(metric_name) if kwargs.get(metric_name) else {})

    def evaluate(self, evaluation_metrics, 
                 kwargs={
                     'accuracy': {'annot':True}
                 }):
        """Evaluate on the train, validation, and test sets.
        """
        self.net.eval()
        self.evaluation_data = {
            'target2ind':self.target2ind,
            'ind2target':self.ind2target,
            'dataset_names': ['Train', 'Validation', 'Test']
        }
        self._instantiate_metrics(evaluation_metrics)
        for data_group in [['Train', self.trainloader],
                           ['Validation', self.valloader],
                           ['Test', self.testloader]]:
            dataset_name, data = data_group
            print(f'Evaluating {dataset_name} set...')
            for input_data, targets in tqdm(data, file=sys.stdout):
                predictions = self.predict(input_data)
                for prediction, target in zip(predictions, targets):
                    self._call_metrics(kwargs, predicted=prediction, truth=target.item(), 
                                       dataset_name=dataset_name, method='update')
            self._call_metrics(kwargs, dataset_name=dataset_name, method='report')
            print()

        self._call_metrics(kwargs, method='aggregate')
        self._call_metrics(kwargs, method='plot')