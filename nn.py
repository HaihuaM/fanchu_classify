import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils import data
import os, argparse
import os.path as op
import numpy as np

class Dataset(data.Dataset):

    def __init__(self, train_x, y):
        self.train_x = train_x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.train_x[index]
        y = self.y[index]
        return X,y

class Net(nn.Module):

    def __init__(self, w_width):
        super(Net, self).__init__()
        self.classfier = nn.Sequential(
                nn.Linear(w_width, 512, bias=False),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                # nn.Linear(1024, 1024),
                # nn.BatchNorm1d(10),
                nn.Linear(512, 10),
                nn.BatchNorm1d(10),
                nn.Linear(10, 10),
                # nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(10, 2)
                )

    def forward(self, x):
        # import pdb; pdb.set_trace();
        feat_max = x.new_tensor([3717])
        feat_min = x.new_tensor([1329])

        x = (x - feat_min)/(feat_max-feat_min)
        x = self.classfier(x)

        return F.log_softmax(x, dim=1)


def train(args, model, device, params):
    """
    Training process.
    """
    # Dataloader
    train_loader = data.DataLoader(train_set, **params)
    # optimizer setup
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # every 25 epoch, learning degrade 50%
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)
    # update learning rate.
    scheduler.step()

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # torch.manual_seed(args.seed)
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        # import pdb; pdb.set_trace()
        output = model(data.to(device))
        
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))


def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            # import pdb; pdb.set_trace()
            # print(torch.nonzero(pred.detach()))
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

def test(args, model, device, params):
    # torch.manual_seed(args.seed)
    test_loader = data.DataLoader(val_set, **params)
    test_epoch(model, device, test_loader)


def set_env(w_width):
    os.environ['SQUENCE_WIDTH'] = str(w_width)

def get_width():
    return int(os.getenv('SQUENCE_WIDTH', 20))

def get_data():
    w_width = get_width()
    data_file = 'squence_data_w%s.npz'%w_width
    if op.exists(data_file):
        data = np.load('squence_data_w%s.npz'%w_width)
        feats = data['feats']
        labels = data['labels'].astype(int)
    else:
        # Geneate data cache.
        print("Generate data for w_width=%s" %w_width)
        feats, labels = data_extend(w_width)
        labels = labels.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(feats, labels, test_size=0.33, random_state=42) 
    train_set = Dataset(X_train, y_train)
    val_set = Dataset(X_test, y_test)
    return train_set, val_set


def data_extend(w_width):
    """
    Prepare data by slide a window on original sequence data.
    """
    if not op.exists('data.npz'):
        import sys
        print("Error: data.npz not exists.")
        sys.exit(0)

    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        
    data = np.load('data.npz')
    feats = data['feats']
    labels = data['labels']

    extend_feats = rolling_window(feats, w_width)
    extend_labels = rolling_window(labels, w_width).mean(axis=1)
    extend_labels = np.rint(extend_labels)

    np.savez_compressed('squence_data_w%s'%w_width, feats = extend_feats, labels = extend_labels)
    return extend_feats, extend_labels

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--w-width', type=int, default=36, metavar='N',
                    help='Width for slidding windown apply on the squence data (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')

if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # w_width = 50
    set_env(args.w_width)

    # Prepare data.
    train_set, val_set = get_data()

    model = Net(args.w_width).double().to(device)

    # Parameters
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 6}

    train(args, model, device, params)

    # Test the model post training.
    test(args, model, device, params)