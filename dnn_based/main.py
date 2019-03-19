from prepare_dataset import prepareDataset
from sample_dataset import musDB_class_dataset
from model import UNet, MLP, DCNN, resnet18, resnet152, resnet34
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import librosa as lsa
import argparse
import os
import time
musdb18_path = '/../musdb18/' #path to the dataset folder
logpath = '/../' #path where we will store the logs



#Parse input arguments
parser = argparse.ArgumentParser()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('ename', help='experiment name, subfolder in '+logpath, type=str)
parser.add_argument('lr', help='learning rate (float)', type=float)
parser.add_argument('tcontext', help='amount of temporal frames for utterance', type=int)
parser.add_argument('num_epochs', help='number of epochs we train the model', type=int)
parser.add_argument('batchnorm', help='bool', type=str)
parser.add_argument('split', help='ratio of data to split into training set, e.g. 0.8', type=float)
parser.add_argument('dropout', help='to use ore not dropout(bool)', type=str)
parser.add_argument('model', help='choose between "unet", "resnet18", "resnet34", "resnet152", "dcnn" or "MLP"', type=str)
parser.add_argument('augment', help='to randomly flip the spectrograms for data augmentation (bool)', type=str)
args=parser.parse_args()

########## EXPERIMENT PARAMETERS Override ##############################################################################
tcontext = args.tcontext # amount of temporal frames considered by the model
fft_size = 1024
dropout=str2bool(args.dropout)
experiment_name = args.ename
lr = args.lr
num_epochs = args.num_epochs#1000
batchnorm = str2bool(args.batchnorm)#'std'  # { 'std', 'meanstd', none' }
split = args.split#0.9 # ratio of the dataset assigned to the training set (rest for validation)
model = args.model
augment = args.augment
########################################################################################################################


if __name__ == "__main__":
    print('<'+experiment_name+'> begins...')
    # Firstly we store a .txt
    if not os.path.exists(logpath + experiment_name):
        os.makedirs(logpath + experiment_name)
    if not os.path.exists(logpath + experiment_name + '/' + 'parameters'):
        os.makedirs(logpath + experiment_name + '/' + 'parameters')

    hyperparams = {'ename': experiment_name, 'tcontext': tcontext, 'fft_size': fft_size, 'lr': lr,
                   'batchnorm': batchnorm, 'split': split, 'dropout':dropout, 'model':model, 'augment':augment}
    print(hyperparams)
    f = open(logpath + experiment_name + '/' + experiment_name + '_hyperparameters.txt', 'w')
    f.write(str(hyperparams))
    f.close()

    # Compute STFs
    print('Preparing dataset...')
    prepareDataset(tcontext, fft_size, musdb18_path)
    print('Dataset prepared.')

    # Load model
    if model=='unet':
        net = UNet(tcontext=tcontext, fft_size=fft_size, dropout=dropout, batchnorm=batchnorm)
    elif model=='mlp':
        net = MLP(tcontext=tcontext, fft_size=fft_size, dropout=dropout)
    elif model=='dcnn':
        net = DCNN(tcontext=tcontext, fft_size=fft_size, dropout=dropout)
    elif model == 'resnet18':
        net = resnet18(tcontext=tcontext, fft_size=fft_size, dropout=dropout, batchnorm=batchnorm)
    elif model == 'resnet152':
        net = resnet152(tcontext=tcontext, fft_size=fft_size, dropout=dropout, batchnorm=batchnorm)
    elif model == 'resnet34':
        net = resnet34(tcontext=tcontext, fft_size=fft_size, dropout=dropout, batchnorm=batchnorm)
    else:
        print('NO MODEL CHOSEN')

    # Load MUSDB dataset classes
    train_dataset = musDB_class_dataset(musdb18_path + 'musdb_classify.hdf5', tcontext,
                                        split, mode='train', augment=augment)

    val_dataset = musDB_class_dataset(musdb18_path + 'musdb_classify.hdf5', tcontext,
                                      split, mode='val', augment=augment)

    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=4, pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=30, shuffle=False, num_workers=4, pin_memory=False, drop_last=True)

    if torch.cuda.is_available():
        net.cuda()
        print('Model sent to GPU')
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    if model == 'resnet18' or model == 'resnet34' or model == 'resnet152':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_loss = torch.zeros(num_epochs)
    val_loss = torch.zeros(num_epochs)
    val_accuracy = torch.zeros(num_epochs)

    print('Training starts...')
    best_acc = 0.0
    val_stucks = 0
    for epoch in range(num_epochs):
        start=time.time()
        running_loss = 0.0
        for x in train_loader:
            data = x['input']
            label = x['label'].squeeze()
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, label)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        train_loss[epoch] = running_loss / (len(train_loader) * train_loader.batch_size)

        correct = 0
        total = 0
        running_loss = 0
        for x in val_loader:
            # disable dropout
            net.eval()
            with torch.no_grad():
                data = x['input']
                label = x['label'].squeeze()
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                outputs = net(data)
                loss = criterion(outputs, label)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        val_loss[epoch] = running_loss / (len(val_loader) * val_loader.batch_size)
        val_accuracy[epoch] = 100 * correct / total
        if val_accuracy[epoch] > best_acc:
            best_acc = val_accuracy[epoch]
            torch.save(net.state_dict(), logpath + experiment_name + '/' + experiment_name + '_params_epoch'+
                       str(epoch + 1).zfill(4) + '.pt')
            val_stucks = 0

            print('Parameters saved.')
        else: #if accuracy has not improved for 3 epochs, we divide learning rate by 10
            val_stucks += 1
            if val_stucks > 4:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/10.0
                    print('Learning Rate decreased by a factor of 10')
                val_stucks = 0

        # re-enable dropout
        net.train()
        print('Epoch ' + str(epoch + 1) + '. Loss: ' + '%.4f' % train_loss[epoch].numpy() + ' (train) | ' + '%.4f' %
              val_loss[epoch].numpy() + ' (val). ACC: ' + '%.4f' % val_accuracy[epoch].item() + '%'+ ' | took '
              + '%.2f' % (time.time()-start) + 's.')



        #Plot a figure with the training curves
        if epoch >=4:
            plt.figure(0)
            plt.plot(train_loss.numpy()[0:epoch], label='train loss')
            plt.plot(val_loss.numpy()[0:epoch], label='val loss')
            plt.legend()
            plt.grid(True)
            plt.xlabel('Epochs')
            plt.ylabel('Cross Entropy')
            plt.savefig(logpath + experiment_name + '/' + experiment_name + '_curves', dpi=200)
            plt.close(0)
            plt.figure(1)
            plt.plot(val_accuracy.numpy()[0:epoch])
            plt.xlabel('Epochs')
            plt.ylabel('Validation Accuracy')
            plt.savefig(logpath + experiment_name + '/' + experiment_name + '_accuracy', dpi=200)
            plt.close(1)

        #save curves for future plotting
        np.save(logpath + experiment_name + '/' + experiment_name +'_accuracy', val_accuracy.numpy())
        np.save(logpath + experiment_name + '/' + experiment_name +'_valloss', val_loss.numpy())
        np.save(logpath + experiment_name + '/' + experiment_name +'_trainloss', train_loss.numpy())

    print('Training ended.')

#Fix imports' warnings in Pycharm:
__all__ = ['torch', 'np', 'nn', 'DataLoader',
           'plt', 'musDB_class_dataset', 'prepareDataset', 'os', 'lsa', 'argparse', ]
