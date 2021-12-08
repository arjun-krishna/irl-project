from numpy.core.defchararray import center
from numpy.core.numerictypes import obj2sctype
from numpy.lib.npyio import save
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from agents.models import ContextPredictionModel
from core.dataset import ContextPredictionDataset
import numpy as np
from core.model_metrics import ModelMetrics
import os
from time import time
import gym
from PIL import Image

from agents.net import Encoder, MLP
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

def train_model(model, train_loader, model_name: str,num_epochs: int = 20, lr=1e-4, save_path: str = './metrics/experiment.pickle', device=torch.device('cuda')):
    loss_fn = nn.CrossEntropyLoss()
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    logger = ModelMetrics(model_name)
    total_iters = 0
    model.train()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        avg_epoch_loss = 0
        for i, (input, label) in enumerate(train_loader):
            center_patch = input['center']
            random_patch = input['random']
            # center_loc = input['center_loc'][0].numpy()
            # random_loc = input['random_loc'][0].numpy()
            # p = input['patch_size'][0].item()
            # half_patch_size = np.array([p//2, p//2])
            # rect_center = patches.Rectangle(tuple(center_loc - half_patch_size),p,p, linewidth=1, edgecolor='b', facecolor='none')
            # rect_random = patches.Rectangle(tuple(random_loc - half_patch_size),p,p, linewidth=1, edgecolor='y', facecolor='none')

            # print('Center Loc', center_loc)
            # print('Random Loc', random_loc)
            observation = input['obs']
            prev_a = input['prev_a']
            label_cp = label['cp']
            label_bc = label['bc']
            # print('Label CP', label_cp[0])
            # f, (a0, a1, a2) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 1]})
            # a0.imshow(observation[0].permute(1,2,0))
            # a0.add_patch(rect_center)
            # a0.add_patch(rect_random)
            # plt.title('Label CP: ' + str(label_cp[0].item()) + '  Label BC: ' + str(label_bc[0].item()))
            # a1.imshow(center_patch[0].permute(1,2,0))
            # a2.imshow(random_patch[0].permute(1,2,0))
            # plt.tight_layout()
            # plt.show()
            # break
            center_patch_features = model.encoder(center_patch)
            random_patch_features = model.encoder(random_patch)
            features = torch.cat([center_patch_features, random_patch_features], dim=1)
            prediction = model.mlp_cp(features)
            image_features = model.encoder(observation)
            # print(image_features.shape)
            # print(prev_a.shape)
            image_features_with_action = torch.cat((image_features, prev_a), dim=1)
            # print(image_features_with_action.shape)
            predicted_action = model.mlp_bc(image_features_with_action)
            # print('Prediction: ', prediction.dtype)
            # print('Label: ', label.reshape(-1).dtype)
            # print('Action Prediction: ', predicted_action.dtype)
            # print('Action Label: ', action_gt.reshape(-1).dtype)
            loss_cp = loss_fn(prediction, label_cp.reshape(-1)).to(device)
            loss_bc = loss_fn(predicted_action, label_bc.reshape(-1)).to(device)
            loss = loss_cp + loss_bc
            logger.add_loss(loss)
            avg_epoch_loss += loss
            loss.backward()
            optimizer.step()
            total_iters += 1
            if i%10==0:
                print('Loss = {}'.format(loss))

        avg_epoch_loss /= (i+1)
        if (epoch+1)%num_epochs == 0:
            model.eval
        print(50*'=')
        print('Epoch {}: , Avg Loss: {}'.format(epoch, avg_epoch_loss))
        with open(save_path, 'wb') as f:
            pickle.dump(logger.getDict(), f)
        if (epoch+1) % args.eval_epoch == 0:
            eval_result = evaluateInEnv()
            logger.add_eval(epoch, eval_result) 


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Parse Command Line Arguments
    parser.add_argument('--num_epochs',default=20, type=int, help='number of epochs to train the model for')
    parser.add_argument('--lr',default=1e-4, type=float, help='learning rate for the model')
    parser.add_argument('--num_demos',default='all', type=int, help='number of demonstrations to use for training model. Use "all" (default) for all demos. Append "m" to integer to train with n_demos in multiples of  that integer. eg: "50m" trains the model with 50, 100, 150, 200, ... demos ')
    parser.add_argument('--input_size', type=int, default=96, help='resize all input images to this size')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--eval_every', type=int, default=5, help='evaluate after every "eval_every" epochs')
    parser.add_argument('--data_root', type=str, default='C:/Users/sanje/Documents/Projects/irl-project/demos/')
    parser.add_argument('--env_name', type=str, default='MiniWorld-Hallway-v0',help='Name of environment: MiniWorld-Hallway-v0 or MiniWorld-YMaze-v0 or MiniWorld-OneRoom-v0 or MiniWorld-FourRooms-v0')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    args = parser.parse_args()

    # Set up experiment details
    model_name = 'ContextPredictionModel'
    model = ContextPredictionModel()
    view = 'top' if args.top_view else 'agent'
    data_folder_path = os.path.join(args.dataroot, args.env_name, view)
    input_shape = (args.input_size, args.input_size)
    train_dataset = ContextPredictionDataset(demo_folder = args.demo_folder, input_shape=input_shape, nb_demos=args.num_demos)
    print('len:',len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    experiment_details = {
        'model_name': model_name,
        'type': 'first/normal',
        'num_demos': args.num_demos,
        'lr': args.lr,
        'view': view,
        'Env':args.env_name
    }
    save_path = './experiments/exp_' + str(time()) + '.pickle'
    best_weights = train_model(model, train_loader, num_epochs=args.num_epochs, model_name=model_name, device = device, save_path=save_path)

    def evaluateInEnv(model):
        model.eval()
        env = gym.make(args.env_name)
        env.reset()
        get_obs = env.render_top_view if args.top_view else env.render_obs

        if args.domain_rand:
            env.domain_rand = True

        NUM_EPISODES = 100

        metric_steps = []
        metric_success = []

        for i in range(NUM_EPISODES):
            done = False
            steps = 0
            while not done:
                x = Image.fromarray(get_obs())
                inp = data_transform(x)
                output = encoder(inp[np.newaxis, :, :, :])
                output = mlp(output)
                output = output.clone().detach()
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
        encoder.train()
        mlp.train()
        return d

    weights = [1., 1., 0.1, 0.01, 0.01, 0.01, 0.01, 0.01] # moving forward is quite likely
    class_weights = torch.FloatTensor(weights)
    cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
    params = list(encoder.parameters()) + list(mlp.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    model_metrics = ModelMetrics(MODEL_NAME)

    encoder.train()
    mlp.train()

    for epoch in range(args.nb_epochs):
        epoch_loss = []
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = encoder(inputs)
            outputs = mlp(outputs)

            loss = cross_entropy(outputs, labels)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_loss = np.mean(epoch_loss)
        print('[%d] loss: %.3f' % (epoch, avg_loss))
        
        model_metrics.add_loss(avg_loss)

        if (epoch+1) % args.eval_epoch == 0:
            eval_result = evaluateInEnv()
            model_metrics.add_eval(epoch, eval_result) 
        
    eval_result = evaluateInEnv()
    model_metrics.add_eval(-1, eval_result) 

    torch.save({
        'encoder_dict': encoder.state_dict(),
        'mlp_dict': mlp.state_dict()
    }, MODEL_PATH)

    with open(MODEL_METRICS_PATH, 'wb') as fp:
        pickle.dump(model_metrics.getDict(), fp)