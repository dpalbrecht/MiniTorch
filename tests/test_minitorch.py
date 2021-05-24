import os
import sys
sys.path.append('..')
import shutil
from minitorch.minitorch import MiniTorch
from minitorch import datasets
import pytest
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# TODO: Test shapes of datasets, etc.
def test_image_classifier():
    """Test 2D (flat) image classifier instantiation, loading, training, and evaluation.
    """
    
    # Instantiate MiniTorch
    num_classes = 2
    X = np.random.rand(100,28,28)
    y = np.array(['0']*50 + ['1']*50)
    minitorch = MiniTorch(X, y,
                          datasets.MatrixDataset,
                          transforms={},
                          val_size=0.10, test_size=0.10,
                          batch_sizes=(10,-1,-1), num_workers=(0,0,0),
                          preprocess=True)
    
    # Define model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, num_classes)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    # Load data into model
    minitorch.load_net(net=Net(), weights='equal',
                       chosen_criterion=nn.CrossEntropyLoss, 
                       chosen_optimizer=optim.SGD,
                       chosen_optimizer_params={'lr':0.01, 'momentum':0.5})
    
    # Train model
    minitorch.train(epochs=1, log_mini_batches=2)
    
    # Load trained model (optional)
    model_checkpoint_dir = '../../../../../model_checkpoints'
    checkpoints_path = f"{model_checkpoint_dir}/{sorted(os.listdir(model_checkpoint_dir))[-1]}/"
    minitorch = MiniTorch.load_checkpoint(net=Net(), 
                                          checkpoints_path=checkpoints_path,
                                          model_name='epoch1_model.pt', load_type='eval')
    shutil.rmtree(checkpoints_path) # cleanup
    
    # Evaluate model
    minitorch.evaluate(evaluation_metrics=['accuracy', 'hit@n'])