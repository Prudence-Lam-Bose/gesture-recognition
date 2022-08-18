"""
Main program for training deep models on capsense data 

TODO:
    Implement MLSTM-FCN in Pytorch
    Get DTW/distance metrics to work 

    If time: 
        Implement attention-based RNN
        Try transformer 
"""

import argparse
import json
import math
import os
import pandas as pd
import torch
import numpy as np 
import utils
import matplotlib.pyplot as plt

from torch.profiler import profile, record_function, ProfilerActivity
from data.dataset import CapsenseDataset
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from modules.lstm_gru import BaseRNN, CRNN, LSTMfcn
from modules.mlstm_fcn import MLSTMfcn
from modules.feature_extractor import FeatureExtractor

from tensorflow.python.keras.layers import Dense, LSTM, Conv1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

# general args
parser = argparse.ArgumentParser(description='Train models on capsense data')
parser.add_argument('-a', '--arch', default='lstm', choices=['rnn','lstm','gru','mlstm-fcn','crnn','lstm-fcn']) # add more models later, this doesn't work
parser.add_argument('--form', '--ff', default='smalls', help='data collection form factor (smalls or goodyear)', choices=["smalls"]) # add goodyear later
parser.add_argument('--lr', '--learning-rate', default=8e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', default=[600, 900], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', default=True, action=argparse.BooleanOptionalAction, help='use cosine lr schedule')
parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--sequence_length', default=15, help='partition time series')
parser.add_argument('--debug', default=False, action=argparse.BooleanOptionalAction, help='use to print model cpu usage')
parser.add_argument('--classical', default=False, action=argparse.BooleanOptionalAction, help='set to true to evaluate on classical models')
parser.add_argument('--classifier', default='knn', choices=['knn','rf','svc'])
parser.add_argument('--num_neighbors', default=None, help='number of neighbors to use for knn if selected')
# for rnn variants 
parser.add_argument('--hidden_dim', default=64, help='size of hidden state')
parser.add_argument('--dropout', default=0.1, help='dropout probability')
parser.add_argument('--num_layers', default=1, type=int)
# utils
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='./results', type=str, metavar='PATH', help='path to cache (default: none)')

global it
it = 0 

def main():
    args = parser.parse_args() # running in cmd 
    
    if args.results_dir == '':
        args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + args.arch 

    # check for gpu availability. 
    is_cuda = torch.cuda.is_available()
    if is_cuda: # if gpu available, set to gpu. otherwise, set to cpu
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # run either classical or deep models
    if args.classical:
        df = utils.preprocess_data(args.form)
        score = utils.classify_with_feature_stats(df, args.classifier, args.num_neighbors) # uses a different feature engineering pipeline
    else: 
        df = utils.preprocess_data(args.form, scale=False)

        ## plot graphs
        # utils.plot_sensor_data(df)
        
        data = CapsenseDataset(df, sequence_length=args.sequence_length)
        
        # create train/test split 
        train_idx, test_idx, _, _ = train_test_split(
            range(len(data)),
            data.labels,
            stratify=data.labels,
            test_size=0.2,
            random_state=42
        )

        train_split = Subset(data, train_idx)
        train_loader = DataLoader(train_split, 
                                batch_size=args.batch_size, 
                                shuffle=True,
                                drop_last=True)

        test_split = Subset(data, test_idx) 
        test_loader = DataLoader(test_split,
                                batch_size = args.batch_size,
                                drop_last=True)

        # create model 
        if args.arch == 'mlstm-fcn':
            model = MLSTMfcn(num_classes=len(set(df.loc[:,"Label"].values)),
                            num_features=data.features.shape[1])
        elif args.arch == 'crnn': 
            model = CRNN(num_features=data.features.shape[1],
                         output_size=len(set(df.loc[:,"Label"].values)),
                         hidden_size=args.hidden_dim)
        elif args.arch == 'lstm-fcn':
            model = LSTMfcn(num_features=data.features.shape[1],
                            output_size=len(set(df.loc[:,"Label"].values)),
                            hidden_size=args.hidden_dim)
        else:
            model = BaseRNN(num_sensors=data.features.shape[1],             # num features
                            hidden_size=args.hidden_dim,                    # size of hidden state vector
                            sequence_length=args.sequence_length,           # window size
                            dropout=args.dropout,                           # dropout probability
                            device=device,                                  # gpu num
                            output_size=len(set(df.loc[:,"Label"].values)), # num classes
                            batch_size=args.batch_size,                     # batch size 
                            num_layers = args.num_layers,                   # num recurrent layers
                            arch=args.arch)                                 # rnn architecture

        model.to(device)

        # define optimizer + criterion
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        # load model if resume
        if args.resume != '':
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print('Loaded from: {}'.format(args.resume))

        # logging
        results = {'train_loss': [], 'test_acc@1': []}
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        # dump args
        with open(args.results_dir + '/args_' + args.arch + '.json', 'w') as fid:
            json.dump(args.__dict__, fid, indent=2)

        # training loop
        train_loss_vals, test_loss_vals = [], [] 
        for epoch in range(args.start_epoch, args.epochs + 1):
            y_train = train_loader.dataset.dataset.labels
            train_loss = train(model, train_loader, optimizer, criterion, epoch, args, y_train, device)
            train_loss_vals.append(train_loss)
            results['train_loss'].append(train_loss)
            test_acc_1, test_loss = test(model, test_loader, criterion, epoch, args, device)
            results['test_acc@1'].append(test_acc_1)
            test_loss_vals.append(test_loss)
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(args.start_epoch, epoch + 1))
            data_frame.to_csv(args.results_dir + '/log_' + args.arch + '.csv', index_label='epoch')
            # save model
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last_' + args.arch + '.pth')

        plot_loss(train_loss_vals, test_loss_vals, args)

def train(net, data_loader, train_optimizer, criterion, epoch, args, labels, device):
    """
    Trains for one epoch 
    """
    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for inputs, labels in train_bar:
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        
        if args.debug: # calculate output with benchmarking for first epoch
            global it 
            if not it:
                with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
                    with record_function('model_inference'):
                        output = net(inputs)
                print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))
                it += 1
            else:
                output = net(inputs) # forward pass 
        else: 
            output = net(inputs)

        # calculate loss 
        loss = criterion(output, labels)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num

def test(net, test_loader, criterion, epoch, args, device):
    """
    Evaluation code for recording top-1 accuracy and loss per epoch
    """
    total_top1, total_num, test_bar, total_loss = 0.0, 0, tqdm(test_loader), 0.0
    for inputs, labels in test_bar:
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        output = net(inputs)
        test_output = output.argmax(1)

        total_num += inputs.size(0) # batch size

        total_top1 += (test_output==labels).sum().item()

        test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

        # calculate loss for plotting
        loss = criterion(output, labels)
        total_loss += loss.item() * test_loader.batch_size 
        epoch_loss = total_loss / total_num

    return total_top1 / total_num * 100, epoch_loss

def extract_features(net, train_loader, device):
    feature_model = FeatureExtractor(net)
    train_bar = tqdm(train_loader)
    features = []

    for inputs, labels in train_bar:
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.no_grad():
            feature = feature_model(inputs)
            features.append(feature.cpu().detach().numpy().reshape(-1))
    features = np.array(features)

    return features 

def knn_with_deep_features():
    pass 

# cosine annealing lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """
    Decay the learning rate based on schedule
    """
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def plot_loss(train_loss, test_loss, args):
    """
    Helper function to plot loss curves
    """
    plt.plot(np.arange(1, args.epochs+1), train_loss, label="Train Loss")
    plt.plot(np.arange(1, args.epochs+1), test_loss, label="Test Loss")
    plt.title("Loss curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig("./loss_curves_" + args.arch)

if __name__ == "__main__":
    main()