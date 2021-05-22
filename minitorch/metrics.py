from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsLookup:
    def __init__(self):
        self.lookup = {
          'accuracy': Accuracy,
          'hit@n': HitAtN
        }

class HitAtN:
    def __init__(self, evaluation_dict):
        self.evaluation_dict = evaluation_dict
        self.dataset_names = self.evaluation_dict['dataset_names']
        self.ind2target = self.evaluation_dict['ind2target']
        self.micro_hitn_dict = {dataset_name: \
                                    {t:{'count':0, 'hit':0} for t in self.evaluation_dict['target2ind']} \
                                        for dataset_name in self.dataset_names}
        self.count = {dataset_name:0 for dataset_name in self.dataset_names}
        self.hit = {dataset_name:0 for dataset_name in self.dataset_names}

    def update(self, predicted_probs, truth_ind, dataset_name):
        sorted_prediction = np.argsort(predicted_probs).flatten()[::-1]
        hit_at_n = np.where(sorted_prediction == truth_ind)[0][0]
        self.micro_hitn_dict[dataset_name][self.ind2target[truth_ind]]['hit'] += hit_at_n
        self.micro_hitn_dict[dataset_name][self.ind2target[truth_ind]]['count'] += 1
        self.count[dataset_name] += 1
        if self.count[dataset_name] > 1:
            self.hit[dataset_name] *= (self.count[dataset_name]-1)
            self.hit[dataset_name] += hit_at_n
            self.hit[dataset_name] /= self.count[dataset_name]
        else:
            self.hit[dataset_name] = hit_at_n

    def report(self, dataset_name):
        print(f"{dataset_name} Hit@N: {self.hit[dataset_name]:.5f}")

    def aggregate(self):
        for dataset_name in self.dataset_names:
            for count_dict in self.micro_hitn_dict[dataset_name].values():
                if count_dict['count'] > 0:
                    count_dict['hit'] /= count_dict['count']

    def plot(self, plot_kwargs):
        sorted_hit = sorted([[card, hitn_dict['hit']] for card, hitn_dict in self.micro_hitn_dict['Test'].items()], 
                            key=lambda x: x[1])
        max_hit_lim = int(np.ceil(max([hitn_dict['hit'] for hitn_dict in self.micro_hitn_dict['Test'].values()]))+1)
        figure = plt.figure(figsize=(max(max_hit_lim, 5), 7))
        plt.bar(range(len(sorted_hit)), [sp[1] for sp in sorted_hit], **plot_kwargs)
        plt.xticks(range(len(sorted_hit)), [sp[0] for sp in sorted_hit], rotation=90)
        for n, group in enumerate(sorted_hit):
            card, hit = group
            plt.text(n, hit, f"{hit:.0f}", ha='center', va='bottom')
        plt.title(f'Average Test Set Hit @ N by Target Class\nMax is {len(sorted_hit)}')
        plt.xlabel('Target Class')
        plt.ylabel('Hit @ N')
        plt.show();

class Accuracy:
    def __init__(self, evaluation_dict):
        self.evaluation_dict = evaluation_dict
        self.dataset_names = self.evaluation_dict['dataset_names']
        self.confusion_matrix_dict = {dataset_name:np.zeros((len(self.evaluation_dict['target2ind']), 
                                                            len(self.evaluation_dict['target2ind']))) \
                                          for dataset_name in self.dataset_names}

    def update(self, predicted_probs, truth_ind, dataset_name):
        sorted_prediction = np.argsort(predicted_probs).flatten()[::-1]
        predicted_ind = sorted_prediction[0]
        self.confusion_matrix_dict[dataset_name][truth_ind][predicted_ind] += 1

    def report(self, dataset_name):
        accuracy = (self.confusion_matrix_dict[dataset_name].diagonal().sum() / self.confusion_matrix_dict[dataset_name].sum(axis=1).sum()) * 100
        print(f'{dataset_name} Accuracy: {accuracy:.2f}%')

    def aggregate(self):
        self.confusion_matrix_normed_dict = {}
        for dataset_name in self.dataset_names:
            self.confusion_matrix_normed_dict[dataset_name] = deepcopy(self.confusion_matrix_dict[dataset_name])
            for row in self.confusion_matrix_normed_dict[dataset_name]:
                row /= row.sum()

    def plot(self, plot_kwargs):
        figure = plt.figure(figsize=(17,15))
        sns.heatmap(self.confusion_matrix_normed_dict['Test'], cmap='rocket_r', **plot_kwargs)
        plt.title('Micro-Accuracy Test Set Predictions Heatmap')
        ticks = [self.evaluation_dict['ind2target'][i] for i in range(len(self.confusion_matrix_normed_dict['Test']))]
        plt.xticks([r+0.5 for r in range(len(self.confusion_matrix_normed_dict['Test']))], ticks, rotation=90, size=8)
        plt.yticks([r+0.5 for r in range(len(self.confusion_matrix_normed_dict['Test']))], ticks, rotation=0, size=8)
        plt.ylabel('Truth')
        plt.xlabel('Prediction')
        plt.show();