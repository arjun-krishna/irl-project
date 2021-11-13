class ModelMetrics:
    loss = []
    evals = {}
    epoch_metrics = {}

    def __init__(self) -> None:
        pass

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
            'epoch_metrics': self.epoch_metrics
        }
    
    def setDict(self, demo_dict):
        if 'loss' in demo_dict:
            self.loss = demo_dict['loss']
        if 'evals' in demo_dict:
            self.evals = demo_dict['evals']
        if 'epoch_metrics' in demo_dict:
            self.epoch_metrics = demo_dict['epoch_metrics']