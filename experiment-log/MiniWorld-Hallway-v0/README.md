# Dataset creation

- Collected 250 human demonstrations

Create datasets:

```python
from core.utils import create_image_dataset
num_demos = 50
seed = 42
create_image_dataset('MiniWorld-Hallway-v0', 'agent', num_demos=num_demos, seed=seed)
```

- dataset/MiniWorld-Hallway-v0/agent/D50/ (seed=42)
- dataset/MiniWorld-Hallway-v0/agent/D100/ (seed=43)
- dataset/MiniWorld-Hallway-v0/agent/D150/ (seed=44)
- dataset/MiniWorld-Hallway-v0/agent/D200/ (seed=45)
- dataset/MiniWorld-Hallway-v0/agent/D250/ (seed=46) # seed doesn't matter


# Architecture

Input: Image (3, 50, 50)

Encoder(
  (conv1): Conv2d(3, 6, kernel_size=(7, 7), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 10, kernel_size=(7, 7), stride=(1, 1))
  (conv3): Conv2d(10, 16, kernel_size=(7, 7), stride=(1, 1))
  (fc1): Linear(in_features=1600, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
)

MLP(
  (fc1): Linear(in_features=128, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=32, bias=True)
  (fc3): Linear(in_features=32, out_features=8, bias=True)
)

Activation Function: ReLU

## Training decisions

Label distribution
- D50 - {0: 79, 1: 69, 2: 1650}
- D100 - {0: 157, 1: 157, 2: 3138, 3: 2}
- D150 - {0: 237, 1: 238, 2: 5078, 3: 2}
- D200 - {0: 336, 1: 330, 2: 6802, 3: 1}
- D250 - {0: 415, 1: 404, 2: 8369, 3: 2}

Defined a weighted cross-entropy loss as follows, becausing label 2 is quite frequent
- [1., 1., 0.1, 0.01, 0.01, 0.01, 0.01, 0.01]

# Behavior Cloning

- D50 (nb_epochs - 40, eval_epoch - 10, seed - 101)
- D100 (nb_epochs - 40, eval_epoch - 10, seed - 102)
- D150 (nb_epochs - 40, eval_epoch - 10, seed - 101)

## Results

# Moco-v1 BC

- D50 (nb_epochs - 70, eval_epoch - 10, seed - 102)
- D100 (nb_epochs - 100, eval_epoch - 10, seed - 102)
