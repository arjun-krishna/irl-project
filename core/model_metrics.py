import matplotlib.pyplot as plt
import pickle

class ModelMetrics:
    loss = []
    evals = {}
    epoch_metrics = {}

    def __init__(self, name, metrics_path=None) -> None:
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
        self.epoch_metric[epoch] = metrics

    def getDict(self):
        return {
            'loss': self.loss,
            'evals': self.evals,
            'epoch_metrics': self.epoch_metrics,
            'name':self.name
        }
    
    def setDict(self, metric_dict):
        self.x_labels = list(metric_dict['evals'].keys())
        if 'loss' in metric_dict:
            self.loss = metric_dict['loss']
        if 'evals' in metric_dict:
            self.evals = metric_dict['evals']
            self.success_rate = [self.evals[i]['success_rate'] for i in self.evals.keys()]
            self.num_steps = [self.evals[i]['metric_steps'] for i in self.evals.keys()]
        if 'epoch_metrics' in metric_dict:
            self.epoch_metrics = metric_dict['epoch_metrics']

    
    def plotMetrics(self, metrics='all'):
        if metrics == 'all':
            metrics = ['loss', 'num_steps', 'success_rate']

        for metric in metrics:
            save_path='./models/' + self.name + '_'+metric + '.png'
            if metric=='loss':
                plt.plot(self.loss)
                plt.xlabel('Number of Epochs')
                plt.ylabel('Loss')
            elif metric=='num_steps':
                bp = plt.boxplot(self.num_steps, patch_artist=True, medianprops=dict(color="black",linewidth=1.5))
                plt.xticks([1,2,3,4,5,6], [5,10,15,20,25,30])
                plt.xlabel('Number of epochs')
                plt.ylabel('# Steps to  reach target')

            elif metric=='success_rate':
                plt.plot(self.x_labels, self.success_rate)
                plt.xlabel('Number of Epochs')
                plt.ylabel('Success Rate')

            plt.savefig(save_path)
            plt.clf()

