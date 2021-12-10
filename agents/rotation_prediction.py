import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from agents.models import RotationPredictionModel
from core.dataset import RotationPredictionDataset
import numpy as np
from core.model_metrics import Logger
import os
from time import time
import gym
from PIL import Image

from agents.net import Encoder, MLP

def train_model(model, train_loader, loss_fn, optimizer, experiment_details,num_epochs: int = 20, eval_every=5, save_path: str = './metrics/experiment.pickle', device=torch.device('cuda'), print_every=10, initial_weight=0.5, weight_decay=0.05):
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    model_name = experiment_details['model_name']
    use_prev_a = experiment_details['use_prev_a']
    logger = Logger(model_name)
    total_iters = 0
    transform = experiment_details['transform']
    model = model.to(device)
    model.train()
    alpha = torch.tensor(initial_weight).to(device)

    for epoch in range(num_epochs):
        avg_epoch_loss = 0
        for i, (input, label) in enumerate(train_loader):

            img1 = input['img1'].to(device)
            img2 = input['img2'].to(device)
            prev_a = input['prev_a']
            label_rp = label['rp']
            label_bc = label['bc']

            # ------- Load all variables from input dict and shift to device -------
            prev_a = input['prev_a'].to(device)
            label_rp = label['rp'].to(device)
            label_bc = label['bc'].to(device)
            # ----------------------------------------------------------------------

            img1_features = model.encoder(img1)
            img2_features = model.encoder(img2)
            features = torch.cat([img1_features, img2_features], dim=1)
            prediction = model.mlp_rp(features)
            # print(image_features.shape)
            # print(prev_a.shape)
            image_features_with_action = torch.cat((img1_features, prev_a), dim=1)
            # print(image_features_with_action.shape)
            predicted_action = model.mlp_bc(image_features_with_action)
            # print('Prediction: ', prediction.dtype)
            # print('Label: ', label.reshape(-1).dtype)
            # print('Action Prediction: ', predicted_action.dtype)
            # print('Action Label: ', action_gt.reshape(-1).dtype)
            loss_rp = loss_fn(prediction, label_rp.reshape(-1)).to(device)
            loss_bc = loss_fn(predicted_action, label_bc.reshape(-1)).to(device)
            loss = alpha*loss_rp + (1-alpha)*loss_bc
            alpha = alpha - weight_decay
            optimizer.zero_grad()
            logger.log_metric('loss_bc', total_iters, loss_bc)
            logger.log_metric('loss_rp', total_iters, loss_rp)
            logger.log_metric('loss_total', total_iters, loss)
            avg_epoch_loss += loss
            loss.backward()
            optimizer.step()
            total_iters += 1
            if i%print_every==0:
                print('RP Loss = {}, BC Loss = {}, Total Loss: {}'.format(loss_rp, loss_bc, loss))

        avg_epoch_loss /= (i+1)

        print('Epoch {}: , Avg Loss: {}'.format(epoch, avg_epoch_loss))
        print(50*'=')

        if eval_every != 0:
            if (epoch+1) % eval_every == 0:
                eval_result = model.eval_in_env(experiment_details['env_name'], device=device, transform=transform, top_view=(experiment_details['view']=='top'), num_episodes=100)
                print('Success rate: ', eval_result['success_rate'], '    Steps: ', eval_result['metric_steps'])
                logger.log_metric('Success rate', epoch, eval_result['success_rate'])
                logger.log_metric('Steps', epoch, eval_result['metric_steps'])
            d = logger.getDict()
            d['experiment_details'] = experiment_details
            d['model_state_dict'] = model.state_dict
            torch.save(d, save_path)

            # if (epoch+1) % eval_every == 0:
            #     eval_result = model.eval_in_env(experiment_details['env_name'], device=device, transform=transform, top_view=(experiment_details['view']=='top'))

            # logger.add_eval(epoch, eval_result) 

    print(' ================================  Evaluating final model 5 times ===============================================')
    for eval_iter in range(5):
        print('------------------------------------------ Eval ',eval_iter, ' ----------------------------------' )
        eval_result = model.eval_in_env(experiment_details['env_name'], device=device, transform=transform, top_view=(experiment_details['view']=='top'), num_episodes=50)
        print('Success rate: ', eval_result['success_rate'], '    Steps: ', eval_result['metric_steps'])
        logger.log_metric('Final Eval Success rate', eval_iter + 1, eval_result['success_rate'])
        logger.log_metric('Final Eval Steps', eval_iter + 1, eval_result['metric_steps'])
        d = logger.getDict()
        d['experiment_details'] = experiment_details
        d['model_state_dict'] = model.state_dict
        torch.save(d, save_path)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Parse Command Line Arguments
    parser.add_argument('--num_epochs',default=20, type=int, help='number of epochs to train the model for')
    parser.add_argument('--lr',default=1e-4, type=float, help='learning rate for the model')
    parser.add_argument('--num_demos',default='all', type=int, help='number of demonstrations to use for training model. Use "all" (default) for all demos. Append "m" to integer to train with n_demos in multiples of  that integer. eg: "50m" trains the model with 50, 100, 150, 200, ... demos ')
    parser.add_argument('--input_size', type=int, default=224, help='resize all input images to this size')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--eval_every', type=int, default=5, help='evaluate after every "eval_every" epochs')
    parser.add_argument('--data_root', type=str, default='C:/Users/sanje/Documents/Projects/irl-project/demos/')
    parser.add_argument('--weighted_ce_loss', action='store_true', help='Use a weighted cross entropy loss')
    parser.add_argument('--env_name', type=str, default='MiniWorld-Hallway-v0',help='Name of environment: MiniWorld-Hallway-v0 or MiniWorld-YMaze-v0 or MiniWorld-OneRoom-v0 or MiniWorld-FourRooms-v0')
    parser.add_argument('--top_view', action='store_true', help='Switch to top view aka world view')
    parser.add_argument('--print_every', type=int, default=30, help='print loss after every "print_every" iterations')
    parser.add_argument('--initial_weight', type=float, default=0.5, help='Initial weight for Context Prediction Loss')
    parser.add_argument('--weight_decay', type=float, default=0, help='Decay weight by this much every epoch')
    parser.add_argument('--use_prev_a', action='store_true', help='Pass previous action as input to the model for predicting next action')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    args = parser.parse_args()

    # Set up experiment details
    model_name = 'RotationPredictionModel'
    model = RotationPredictionModel()
    view = 'top' if args.top_view else 'agent'
    data_folder_path = os.path.join(args.data_root, args.env_name, view)
    input_shape = (args.input_size, args.input_size)
    train_dataset = RotationPredictionDataset(demo_folder = data_folder_path, input_shape=input_shape, nb_demos=args.num_demos, patch_mode='random')
    print('len:',len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    weights = [1., 1., 0.1, 0.01, 0.01, 0.01, 0.01, 0.01] # moving forward is quite likely
    class_weights = torch.FloatTensor(weights)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights) if args.weighted_ce_loss else nn.CrossEntropyLoss()
    params = list(model.encoder.parameters()) + list(model.mlp_rp.parameters()) + list(model.mlp_bc.parameters())
    optimizer = optim.Adam(params=params, lr=args.lr)
    experiment_details = vars(args)
    experiment_details['model_name'] = model_name
    experiment_details['transform'] = train_dataset.transform
    experiment_details['optimizer'] = optimizer.state_dict
    experiment_details['view'] = 'top' if args.top_view else 'agent'
    save_path = './experiments/exp_' + str(time()) + '.pickle'
    
    best_weights = train_model(model, train_loader, loss_fn, optimizer,experiment_details, num_epochs=args.num_epochs, eval_every=args.eval_every, device = device, save_path=save_path, print_every=args.print_every, initial_weight=args.initial_weight, weight_decay=args.weight_decay)