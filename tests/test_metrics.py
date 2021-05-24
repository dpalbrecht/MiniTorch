import os
import sys
sys.path.append('..')
from minitorch import metrics
import subprocess
import pytest
import numpy as np


# TODO: Test Hit@N
def test_accuracy():
    """Test the accuracy, confusion matrix, and micro-accuracy calculations.
    """
    
    preds_dict = {
        'Train': {
            'preds': [[.9,.1,0], [.2,.8,0], [.9,.1,0]],
            'targets': [0,1,2]
        },
        'Validation': {
            'preds': [[.9,.1,0], [.2,.8,0], [.9,.1,0]],
            'targets': [0,0,2]
        },
        'Test': {
            'preds': [[.9,.1,0], [.2,.8,0], [.9,.1,0]],
            'targets': [0,1,0]
        }
    }
    checks_dict = {
        'Train': {
            'accuracy': (2/3)*100,
            'confusion_matrix': np.array([
                [1,0,0],
                [0,1,0],
                [1,0,0]
            ]),
            'confusion_matrix_normed': np.array([
                [1,0,0],
                [0,1,0],
                [1,0,0]
            ])
        },
        'Validation': {
            'accuracy': (1/3)*100,
            'confusion_matrix': np.array([
                [1,1,0],
                [0,0,0],
                [1,0,0]
            ]),
            'confusion_matrix_normed': np.array([
                [0.5,0.5,0],
                [0,0,0],
                [1,0,0]
            ])
        },
        'Test': {
            'accuracy':100,
            'confusion_matrix': np.array([
                [2,0,0],
                [0,1,0],
                [0,0,0]
            ]),
            'confusion_matrix_normed': np.array([
                [1,0,0],
                [0,1,0],
                [0,0,0]
            ])
        }
    }
    
    target2ind = {'class1':0,'class2':1,'class3':2}
    ind2target = {v:k for k,v in target2ind.items()}
    accuracy_metric = metrics.Accuracy({
        'target2ind':target2ind,
        'ind2target':ind2target,
        'dataset_names': ['Train', 'Validation', 'Test']
    })
    
    for dataset_name, pred_group in preds_dict.items():
        for pred, target in zip(pred_group['preds'], pred_group['targets']):
            accuracy_metric.update(pred, target, dataset_name)
        accuracy_metric.report(dataset_name)
        assert checks_dict[dataset_name]['accuracy'] == accuracy_metric.accuracy
        assert (checks_dict[dataset_name]['confusion_matrix'] == accuracy_metric.confusion_matrix_dict[dataset_name]).all()

    accuracy_metric.aggregate()
    for dataset_name in preds_dict.keys():
        assert (checks_dict[dataset_name]['confusion_matrix_normed'] == accuracy_metric.confusion_matrix_normed_dict[dataset_name]).all()