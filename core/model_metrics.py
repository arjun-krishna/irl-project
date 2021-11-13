class ModelMetrics:
    loss = []
    evals = {}

    def __init__(self) -> None:
        pass

    def add_loss(self, l: float):
        self.loss.append(l)
    
    def add_eval(self, epoch, metrics):
        self.evals[epoch] = metrics

    def getDict(self):
        return {
            'loss': self.loss,
            'evals': self.evals
        }
    
    def setDict(self, demo_dict):
        for field in ['loss', 'evals']:
            assert field in demo_dict
        self.observations = demo_dict['loss']
        self.actions = demo_dict['evals']