class Dataset:

    observations = []
    actions = []

    def __init__(self):
        pass
    
    def setDict(self, data_dict: dict) -> None:
        for field in ['obs', 'action']:
            assert field in data_dict
        
        self.observations = data_dict['obs']
        self.actions = data_dict['action']

    def size(self) -> int:
        return len(self.observations)