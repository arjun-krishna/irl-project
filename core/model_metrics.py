import matplotlib.pyplot as plt
plt.style.use('ggplot')
import torch
import pickle
import numpy as np


class Logger:
    loss = []
    evals = {}
    epoch_metrics = {}

    def __init__(self, experiment_details) -> None:
        '''
            name: Model Name
            metrics_path: Path to save metrics dict
        '''
        self.data_dict = {}
        self.data_dict['experiment_details'] = experiment_details
    
    def load_metrics(self, file_path):
        self.data_dict = torch.load(file_path)

    def log_metric(self, name:str, x, y):
        if name not in self.data_dict.keys():
            self.data_dict[name] = {}
        self.data_dict[name][x] = y
            
    def visualize_results(self, save_path, metrics: list = 'all'):
        if metrics == 'all':
            metrics = []
        n_subplots = len(metrics)
        for i, metric in enumerate(metrics):
            plt.subplot(1, n_subplots, i)
            if 'loss' or 'success' in metric:
                x_values = list(self.data_dict[metric].keys())
                y_values = list(self.data_dict[metric].values())
                plt.plot(x_values, y_values)
                plt.title(metric)
            elif 'steps' in metric:
                num_steps = list(self.data_dict[metric].values())
                num_steps = np.array([np.array(xi) for xi in num_steps])
                bp = plt.boxplot(num_steps, patch_artist=True, boxprops=dict(facecolor='#56B4E9'), medianprops=dict(color="black",linewidth=1.5))
                x_values = list(self.data_dict[metric].keys())
                x_locs = range(1, len(x_values) + 1)
                plt.xticks(x_locs, x_values)
                plt.xlabel('Number of epochs')
                plt.ylabel('# Steps to  reach target')
                plt.title(metric)
            plt.savefig(save_path)
            plt.clf()

    def getDict(self):
        return self.data_dict



