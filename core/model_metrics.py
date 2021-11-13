import matplotlib.pyplot as plt
import pickle

class ModelMetrics:
    loss = []
    evals = {}
    epoch_metrics = {}

    def __init__(self, name, metrics_path) -> None:
        self.name = name
        if metrics_path:
            with open(metrics_path, 'rb') as f:
                d = pickle.load(f)
                self.setDict(d)

    def add_loss(self, l: float):
        self.loss.append(l)
    
    def add_eval(self, epoch, metrics):
        self.evals[epoch] = metrics

    def add_epoch_metric(self, epoch, metrics):
        self.epoch_metrics[epoch] = metrics

    def getDict(self):
        return {
            'loss': self.loss,
            'evals': self.evals,
            'epoch_metrics': self.epoch_metrics,
            'name':self.name
        }
    
    def setDict(self, metric_dict):
        if 'name' in metric_dict:
            self.name = metric_dict['name']
        if 'loss' in metric_dict:
            self.loss = metric_dict['loss']
        if 'evals' in metric_dict:
            self.evals = metric_dict['evals']
        if 'epoch_metrics' in metric_dict:
            self.epoch_metrics = metric_dict['epoch_metrics']
    
    def plotLoss(self, ):
        self.name
        save_path='./models/loss.png'
        plt.plot(self.loss)
        plt.savefig(save_path)
