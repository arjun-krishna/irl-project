import torch
import argparse
import random
import os
from pathlib import Path
import numpy as np
from torch import nn
import torch.optim as optim
from torchvision.transforms.transforms import RandomApply
from core.transforms import GaussianBlur, TwoCropsTransform
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from agents.net import MLP, Encoder, SimSiam
import time
from core.hist_dataset import DemoDataPreviousAction
from pathlib import Path

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.style.use('ggplot')

print(os.getpid())

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, default='', type=str, help='image dataset folder to train model from')
parser.add_argument('--demo_path', required=True, default='', type=str, help='numpy demo folder to train model from')
parser.add_argument('--nb_demos', required=True, default=300, type=int, help='numpy demo number of episodes to get')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--nb_epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--seed', default=102, type=int, help='random seed')
parser.add_argument('--gpu', action='store_true', help='Uses a gpu when possible')
parser.add_argument('--top_view', action='store_true', help='what view?')
args = parser.parse_args()
print('Note that demo_path must have agent/ and top/ folders')
view = 'agent' if not args.top_view else 'top'
DATA_PATH = os.path.join(args.data_path)
DEMO_PATH = os.path.join(args.demo_path, view)

device='cpu'
if args.gpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


if not os.path.exists('models/'):
    os.makedirs('models/')
MODEL_PATH = 'models/' + Path(args.data_path).parts[-1] + "_" + view + '_simsiam_bc.pt'

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Needed transformations
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
augmentation = [
    transforms.RandomResizedCrop(50, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.ToTensor(),
    normalize
]

train_dataset = datasets.ImageFolder(
    DATA_PATH,
    TwoCropsTransform(
        transforms.Compose(augmentation)
    )
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

model = SimSiam().to(device)

criterion = nn.CosineSimilarity(dim=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# define some loggers
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

loss_history = []
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    epoch_losses = []
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        p1, p2, z1, z2 = model(x1=images[0].to(device), x2=images[1].to(device))
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images[0].size(0))
        epoch_losses.append(loss.item())
        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (max(len(train_loader)//10, 1)) == 0:
            progress.display(i)
    loss_history.append(sum(epoch_losses)/len(epoch_losses))
for epoch in range(1, 1 + args.nb_epochs):
    train(train_loader, model, criterion, optimizer, epoch=epoch)

# now train classifier
print('Now training classifier with pretrained encoder...')
cls_train_dataset = DemoDataPreviousAction(demo_folder=DEMO_PATH, nb_demos=args.nb_demos, transform=transforms.Compose(augmentation))
cls_train_dataloader = DataLoader(cls_train_dataset,batch_size=args.batch_size,shuffle=True)

mlp_classifier = MLP(dim=2304+9).to(device)
sup_criterion = nn.CrossEntropyLoss().to(device)
sup_opt = optim.Adam(mlp_classifier.parameters(), lr=args.lr)
def grad_model(model, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad

clf_epoch_loss_history = []

# freeze representation, use encoder to get features, then train the FCN
def cls_train(train_loader, model, cls_mlp, criterion, optimizer, epoch):
    enc_model = model.encoder
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    enc_model.train()
    grad_model(enc_model, False)
    cls_mlp.train()

    end = time.time()
    clf_losses = []
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = batch['obs'].to(device)
        prev_action = batch['prev_a'].to(device)
        target = batch['a']
        z = enc_model(images.to(device))
        # print(p1.size(), p2.size(), z1.size(), z2.size())
        z = torch.cat((z, prev_action), dim=-1)
        logits = cls_mlp(z)
        loss = criterion(logits, target.to(device).flatten())

        losses.update(loss.item(), images[0].size(0))
        clf_losses.append(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (max(len(train_loader)//10, 1)) == 0:
            progress.display(i)
    clf_epoch_loss_history.append(sum(clf_losses)/len(clf_losses))
for epoch in range(1, 1 + args.nb_epochs):
    cls_train(cls_train_dataloader,
              model,
              mlp_classifier,
              sup_criterion, sup_opt, epoch=epoch)


print('Saving...')
print(MODEL_PATH)
torch.save({
    'encoder_dict': model.encoder.state_dict(),
    'mlp_dict': mlp_classifier.state_dict()
}, MODEL_PATH)

# log_dir = 
# if not os.path.exists(directory):
#     os.makedirs(directory)

plt.figure()
plt.plot(range(1, args.nb_epochs + 1), loss_history, '-bo')
plt.title('SimSiam Average Loss vs Epoch')
plt.xlabel('epoch')
plt.ylabel('Avg. Loss')
plt.show()
plt.clf()

plt.figure()
plt.plot(range(1, args.nb_epochs + 1), clf_epoch_loss_history, '-bo')
plt.title('SimSiam-trained Classifier Average Loss vs Epoch')
plt.xlabel('epoch')
plt.ylabel('Avg. Loss')
plt.show()
plt.clf()