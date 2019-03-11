from prepare_dataset import prepareDataset
from sample_dataset import musDB_class_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import librosa as lsa
import argparse
import os

musdb18_path = '/home/enricguso/datasets/musdb18/' #path to the dataset folder
logpath = '/media/archive/' #path where we will store the logs



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
parser.add_argument('split', help='ratio of data to split into training set', type=float)
parser.add_argument('dropout', help='to use ore not dropout(bool)', type=str)
parser.add_argument('model', help='choose between "unet" or "MLP"', type=str)
parser.add_argument('augment', help='to randomly flip the inputs (bool)', type=str)
args=parser.parse_args()

########## EXPERIMENT PARAMETERS Override ##############################################################################
tcontext = args.tcontext # amount of temporal frames considered by the model
fft_size = 1024
dropout=str2bool(args.dropout)
experiment_name = args.ename
lr = args.lr#1.0 # learning rate
num_epochs = args.num_epochs#1000
batchnorm = str2bool(args.batchnorm)#'std'  # { 'std', 'meanstd', none' }
split = args.split#0.9 # ratio of the dataset assigned to the training set (rest for validation)
model = args.model
augment = args.augment
########################################################################################################################

class UNet(nn.Module):
    def __init__(self, tcontext, fft_size, dropout, batchnorm):
        print('Using a U-NET')
        self.tcontext = tcontext
        self.fft_size = fft_size
        self.dropout=dropout
        self.batchnorm=batchnorm
        super(UNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (5, 5), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (5, 5), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 5), stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (5, 5), stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, (5, 5), stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, (5, 5), stride=2)
        self.bn6 = nn.BatchNorm2d(512)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.drop4 = nn.Dropout2d(p=0.5)
        self.drop5 = nn.Dropout2d(p=0.5)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.drop7 = nn.Dropout2d(p=0.5)

        #self.fc1 = nn.Linear(512 * 4 * 5, 512)
        self.fc1 = nn.Linear(512 * 4 * 5, 4)

        #self.fc2 = nn.Linear(512, 128)
        #self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn1(x)
        x = self.lrelu(x)
        if self.dropout:
            x = self.drop1(x)
        x = self.conv2(x)
        if self.batchnorm:
            x = self.bn2(x)
        x = self.lrelu(x)
        if self.dropout:
            x = self.drop2(x)
        x = self.conv3(x)
        if self.batchnorm:
            x = self.bn3(x)
        if self.dropout:
            x = self.drop3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        if self.batchnorm:
            x = self.bn4(x)

        x = self.lrelu(x)
        if self.dropout:
            x = self.drop4(x)
        x = self.conv5(x)
        if self.batchnorm:
            x = self.bn5(x)
        if self.dropout:
            x = self.drop5(x)
        x = self.lrelu(x)
        if self.dropout:
            x = self.drop6(x)
        x = self.conv6(x)
        if self.batchnorm:
            x = self.bn6(x)
        if self.dropout:
            x = self.drop7(x)
        x = self.lrelu(x)
        x = x.view(-1, 512 * 4 * 5)
        #x = nn.functional.relu(self.fc1(x))
        #if self.dropout:
        #    x=self.drop4(x)
        #x = nn.functional.relu(self.fc2(x))
        #if self.dropout:
        #    x=self.drop5(x)
        x = (self.fc1(x))

        return x.squeeze()

class MLP(nn.Module):

    def __init__(self, tcontext, fft_size, dropout):
        print('Using a MLP')
        self.tcontext = tcontext
        self.fft_size = fft_size
        self.dropout=dropout
        super(MLP, self).__init__()

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)


        self.fc1 = nn.Linear(self.tcontext*int(self.fft_size/2+1), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):

        x = x.view(-1, self.tcontext*int(self.fft_size/2+1))
        x = nn.functional.relu(self.fc1(x))
        if self.dropout:
            x=self.drop1(x)
        x = nn.functional.relu(self.fc2(x))
        if self.dropout:
            x=self.drop2(x)
        x = self.fc3(x)

        return x.squeeze()


if __name__ == "__main__":
    print('<'+experiment_name+'> begins...')
    # Firstly we store a .txt
    if not os.path.exists(logpath + experiment_name):
        os.makedirs(logpath + experiment_name)
    if not os.path.exists(logpath + experiment_name + '/' + 'parameters'):
        os.makedirs(logpath + experiment_name + '/' + 'parameters')

    hyperparams = {'ename': experiment_name, 'tcontext': tcontext, 'fft_size': fft_size, 'lr': lr,
                   'batchnorm': batchnorm, 'split': split, 'dropout':dropout, 'model':model}
    print(hyperparams)
    f = open(logpath + experiment_name + '/' + experiment_name + '_hyperparameters.txt', 'w')
    f.write(str(hyperparams))
    f.close()

    # Compute STFs
    #print('Preparing dataset...')
    #prepareDataset(tcontext, fft_size, musdb18_path)
    print('Dataset prepared.')

    # Load model
    if model=='unet':
        net = UNet(tcontext=tcontext, fft_size=fft_size, dropout=dropout, batchnorm=batchnorm)
    elif model=='mlp':
        net = MLP(tcontext=tcontext, fft_size=fft_size, dropout=dropout)
    else:
        print('NO MODEL CHOSEN')

    # Load MUSDB dataset classes
    train_dataset = musDB_class_dataset(musdb18_path + 'musdb_classify.hdf5', tcontext,
                                        split, mode='train', augment=augment)

    val_dataset = musDB_class_dataset(musdb18_path + 'musdb_classify.hdf5', tcontext,
                                      split, mode='val', augment=augment)

    train_loader = DataLoader(train_dataset, batch_size=120, shuffle=True, num_workers=4, pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=120, shuffle=False, num_workers=4, pin_memory=False, drop_last=True)

    if torch.cuda.is_available():
        net.cuda()
        print('Model sent to GPU')
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_loss = torch.zeros(num_epochs)
    val_loss = torch.zeros(num_epochs)
    val_accuracy = torch.zeros(num_epochs)

    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch ' + str(epoch + 1) + ' starts...')
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
            print('Parameters saved.')

        # re-enable dropout
        net.train()
        print('Epoch ' + str(epoch + 1) + '. Loss: ' + '%.4f' % train_loss[epoch].numpy() + ' (train) | ' + '%.4f' %
              val_loss[epoch].numpy() + ' (val). ACC: ' + '%.4f' % val_accuracy[epoch].item() + '%')



        #Plot a figure with the training curves
        if epoch >=4:
            plt.figure(0)
            plt.plot(train_loss.numpy()[0:epoch], label='train loss')
            plt.plot(val_loss.numpy()[0:epoch], label='val loss')
            plt.legend()
            plt.grid(True)
            plt.xlabel('epochs')
            plt.ylabel('CEL')
            plt.savefig(logpath + experiment_name + '/' + experiment_name + '_curves', dpi=200)
            plt.close(0)
            plt.figure(1)
            plt.plot(val_accuracy.numpy()[0:epoch])
            plt.xlabel('epochs')
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
