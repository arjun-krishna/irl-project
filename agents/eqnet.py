import torch
from e2cnn import gspaces, nn
import argparse
import pickle
import random
import gym
import gym_miniworld
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from PIL import Image
from agents.hist_dataset import DemoDataPreviousAction
from core.model_metrics import ModelMetrics

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

class EqNet(torch.nn.Module):

    def __init__(self, args):
        super(EqNet, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(8)
        
        in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        self.input_type = in_type
        out32_type = nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        out64_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        out128_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])

        self.blk1 = nn.SequentialModule(
            nn.R2Conv(in_type, out32_type, kernel_size=3, padding=1),
            nn.ReLU(out32_type, inplace=True),
            nn.InnerBatchNorm(out32_type),
            nn.R2Conv(out32_type, out64_type, kernel_size=3, padding=1),
            nn.ReLU(out64_type, inplace=True),
            nn.InnerBatchNorm(out64_type),
            nn.PointwiseMaxPool(out64_type, 2)
        )

        in_type = self.blk1.out_type
        self.blk2 = nn.SequentialModule(
            nn.R2Conv(in_type, out64_type, kernel_size=3, padding=1),
            nn.ReLU(out64_type, inplace=True),
            nn.InnerBatchNorm(out64_type),
            nn.R2Conv(out64_type, out128_type, kernel_size=3, padding=1),
            nn.ReLU(out128_type, inplace=True),
            nn.InnerBatchNorm(out128_type),
            nn.PointwiseMaxPool(out128_type, 2)
        )

        in_type = self.blk2.out_type
        self.blk3 = nn.SequentialModule(
            nn.R2Conv(in_type, out64_type, kernel_size=3, padding=1),
            nn.ReLU(out64_type, inplace=True),
            nn.InnerBatchNorm(out64_type),
            nn.R2Conv(out64_type, out64_type, kernel_size=3, padding=1),
            nn.ReLU(out64_type, inplace=True),
            nn.InnerBatchNorm(out64_type),
            nn.PointwiseMaxPool(out64_type, 2),
            nn.GroupPooling(out64_type)
        )

        self.no_hist = args.no_hist
        out_dim = 4096 + (0 if self.no_hist else 9)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(out_dim, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, 8)
        )

    def forward(self, sample):
        x = nn.GeometricTensor(sample['obs'], self.input_type)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x).tensor
        if self.no_hist:
            x = x.reshape(x.shape[0], -1)
        else:
            x = torch.concat([x.reshape(x.shape[0], -1), sample['prev_a']], dim=1)        
        x = self.fc(x)
        return x

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-OneRoom-v0', type=str, help='env name to evaluate on')
parser.add_argument('--top-view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--data-path', default='', type=str, help='folder of demos')
parser.add_argument('--nb-demos', default=50, type=int, help='number of demos to consider for training')
parser.add_argument('--batch-size', default=64, type=int, help='training batch size')
parser.add_argument('--nb-epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--eval-epoch', default=10, type=int, help='run evaluate after specified epochs')
parser.add_argument('--lr', default=0.01, type=float, help='max learning rate')
parser.add_argument('--seed', default=101, type=int, help='random seed')
parser.add_argument('--no-hist', action='store_true', help='remove history action feature')
args = parser.parse_args()

if not os.path.exists('models/'):
    os.makedirs('models/')
MODEL_NAME = 'D' + str(args.nb_demos) + '_eqnet' + ('_hist' if not args.no_hist else '')
MODEL_PATH = 'models/' + MODEL_NAME + '.pt'
MODEL_METRICS_PATH = 'models/' + MODEL_NAME + '.pickle'

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

### GPU data transfer utils
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if isinstance(data, dict):
        return dict((key, to_device(data[key], device)) for key in data)
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Transfer data to the device and yield the transfered data"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)
## utils end

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
    normalizer
])

device = get_default_device()
print(f'Using device = {device}')

train_dataset = DemoDataPreviousAction(args.data_path, args.nb_demos, data_transform)
train_loader = DeviceDataLoader(DataLoader(train_dataset, args.batch_size, shuffle=True), device)

model = to_device(EqNet(args), device)

def evaluateInEnv():
    env = gym.make(args.env_name)
    env.reset()
    get_obs = env.render_top_view if args.top_view else env.render_obs

    if args.domain_rand:
        env.domain_rand = True

    NUM_EPISODES = 50

    metric_steps = []
    metric_success = []

    for i in range(NUM_EPISODES):
        done = False
        steps = 0
        prev_a = 8
        while not done:
            sample = {
                'obs': torch.unsqueeze(data_transform(Image.fromarray(get_obs())), 0),
                'prev_a': F.one_hot(torch.tensor([prev_a]), num_classes=9)
            }
            output = model(to_device(sample, device))
            output = output.clone().detach().cpu()
            action = output[0].argmax().item()

            _, reward, done, _ = env.step(action)
            steps += 1

            if done:
                env.reset()
                metric_success.append(reward > 0)
                metric_steps.append(steps)
    d = {
        'success_rate': np.mean(metric_success),
        'metric_steps': metric_steps
    }
    return d

weights = [1., 1., 0.6, 0.01, 0.01, 0.01, 0.01, 0.01] # moving forward is quite likely
class_weights = torch.FloatTensor(weights)
cross_entropy = torch.nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.nb_epochs)

model_metrics = ModelMetrics(MODEL_NAME)

if device.type == 'cuda':
    torch.cuda.empty_cache()
model.train()
for epoch in range(args.nb_epochs):
    epoch_loss = []
    for i, sample in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(sample)
        loss = cross_entropy(outputs, sample['a'].squeeze())
        loss.backward()
        epoch_loss.append(loss.detach().cpu().item())
        optimizer.step()
    scheduler.step()
    avg_loss = np.mean(epoch_loss)
    print('[%d] loss: %.3f' % (epoch, avg_loss))
    
    model_metrics.add_loss(avg_loss)

    if (epoch+1) % args.eval_epoch == 0:
        model.eval()
        eval_result = evaluateInEnv()
        print('EVAL (success_rate) = ', eval_result['success_rate'])
        model_metrics.add_eval(epoch, eval_result) 
        model.train()

model.eval()      
eval_result = evaluateInEnv()
model_metrics.add_eval(-1, eval_result) 

torch.save({
    'model_dict': model.state_dict()
}, MODEL_PATH)

with open(MODEL_METRICS_PATH, 'wb') as fp:
    pickle.dump(model_metrics.getDict(), fp)