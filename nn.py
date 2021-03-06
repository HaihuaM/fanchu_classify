import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils import data
import os, sys, argparse, math, pickle
import os.path as op
import numpy as np

class Dataset(data.Dataset):
    """
    Build a data iterator.
    """
    def __init__(self, train_x, y):
        self.train_x = train_x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.train_x[index]
        y = self.y[index]
        return X,y

class Flatten(nn.Module):
    """
    A flatten layer
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        w_width = get_width()
        # calculate the width of data for cnn
        self.kernel_w = int(math.sqrt(w_width))

        # Network setup
        # conv -> batch normalization -> conv -> max pool -> conv -> batch normalization -> fc
        self.classfier_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 1),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 16, 2),
                nn.MaxPool2d(2, 2),
                nn.ReLU(),
                nn.Conv2d(16, 9, 2),
                nn.BatchNorm2d(9),
                Flatten(),
                nn.Linear(w_width, 2),
                nn.ReLU(),
                )

    def forward(self, x):
        # Observered from the data to see the typically max and min
        feat_max = x.new_tensor([3717])
        feat_min = x.new_tensor([1329])
        # normalize the data to 1
        x = (x - feat_min)/(feat_max-feat_min)
        # reshape data to [batch_siae, 1,  kernel_w, kernel_w] for conv2d
        x = x.view(-1, 1, self.kernel_w, self.kernel_w)
        # feed data to cnn  
        x = self.classfier_cnn(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, params):
    """
    Training process.
    """
    # Dataloader
    train_loader = data.DataLoader(train_set, **params)
    # optimizer setup
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001 )
    # every 25 epoch, learning degrade 50%
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5, last_epoch=-1)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)
        # Test the model post training.
        acc = test(args, model, device, params)
        # Save model
        checkpoint_dir = "checkpoints"
        checkpoint = op.join(checkpoint_dir, 'epoch_%s_%.3f.pth'%(epoch, acc))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'w_width': get_width(), 
            },checkpoint)
    # update learning rate.
    scheduler.step()


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # torch.manual_seed(args.seed)
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
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
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    return (correct / len(data_loader.dataset))

def test(args, model, device, params):
    test_loader = data.DataLoader(val_set, **params)
    acc = test_epoch(model, device, test_loader)
    return acc


def set_env(w_width):
    os.environ['SQUENCE_WIDTH'] = str(w_width)

def get_width():
    return int(os.getenv('SQUENCE_WIDTH', 36))

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


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def data_extend(w_width):
    """
    Prepare data by slide a window on original sequence data.
    """
    if not op.exists('data.pickle'):
        import sys
        print("Error: data.pickle not exists.")
        sys.exit(0)
        
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)

    feats = list()
    labels = list()

    for idx, feat in enumerate(data['feats']):
        label = data['labels'][idx]
        feat = rolling_window(feat, w_width)
        feats.append(feat)
        label = np.rint(rolling_window(label, w_width).mean(axis=1))
        labels.append(label)

    labels = np.concatenate(labels, axis=0)
    feats = np.concatenate(feats, axis=0)

    np.savez_compressed('squence_data_w%s'%w_width, feats = feats, labels = labels)
    return feats, labels

def build_inference_data(data_file):
    w_width = get_width()
    if data_file.endswith('.csv'):
        feat  = np.loadtxt(data_file, delimiter=',', usecols=(6,), skiprows=1, unpack=True)
        feat = rolling_window(feat, w_width)
        return feat
    else:
        print("Error: only csv file is support.")

def inference(args, device, 
        checkpoint, input_file):

    checkpoint = torch.load(checkpoint)
    w_width = checkpoint['w_width']
    set_env(w_width)

    model = Net().double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001 )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    feat_data = torch.from_numpy(build_inference_data(input_file))
    out = model(feat_data.to(device))
    pred = out.max(1)[1].to('cpu').detach().numpy()
    file_name = input_file.replace('.csv', '.predict.csv')
    dump_result(pred, input_file, file_name)

def dump_result(pred, data_file, file_name, skiprows=1):    
    with open(data_file, 'r') as f:
        lines = f.readlines()[skiprows:]

    for idx, l in enumerate(pred):
        lines[idx] = lines[idx].strip('\n')
        lines[idx] += ', %s\n' %l

    content = "".join(lines)
    with open(file_name, 'w') as f:
        f.write(content)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=8192, metavar='N',
                    help='input batch size for training (default: 2048)')
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
parser.add_argument('--inference', action='store_true', default=False,
                    help='for inference the result')
parser.add_argument('--input', 
                    help='input csv file for inference')
parser.add_argument('--checkpoint', 
                    help='point the checkpoint pth')

if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    if args.inference:
        if args.checkpoint == None or \
                args.input == None:
            print("Error: Inference mode, you need input the checkpoint pth, and the data file.")
            sys.exit(0)

        if op.exists(args.checkpoint) and \
                op.exists(args.input):
            print("Inference the model.")
            inference(args, device,
                    args.checkpoint, args.input)
            sys.exit(0)
        else:
            print("Error: Pleas ensure %s or %s exists"%(args.checkpoint, \
                    args.input))
            sys.exit(0)
            
    # w_width = 36
    set_env(args.w_width)


    # Prepare data.
    train_set, val_set = get_data()

    model = Net().double().to(device)

    # Parameters
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 6}

    train(args, model, device, params)

