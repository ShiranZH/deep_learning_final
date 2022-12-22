'''Train CIFAR10 with PyTorch.'''
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from models.resnet_new import ResNet34
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from mobile_former import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
net = MobileFormer(config_52)
# net = ResNet34()
net = net.to(device)
writer = SummaryWriter('log')
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
optimizer = optim.AdamW(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.2)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))


        optimizer.zero_grad()
        outputs = net(inputs)
        # loss = criterion(outputs, targets)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        writer.add_scalar('train/train_loss', train_loss, epoch)
        if not batch_idx % 20:
            print('Training Epoch:{} [{}/{}]\tLoss:{:0.4f}\tLR:{:0.6f} '
                  .format(epoch, batch_idx * 128 + len(targets), len(trainloader.dataset),
                          train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))
            train_log_txt.write('Training Epoch:{} [{}/{}]\tLoss:{:0.4f}\tLR:{:0.6f}\n'
                                .format(epoch, batch_idx * 128 + len(targets), len(trainloader.dataset),
                                        train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))
    train_loss_history.append(train_loss / (batch_idx + 1))


def cifar_test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # progress_bar(batch_idx, len(testloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        print('Test Epoch:{} \tLoss:{:0.4f} \tAccuracy:{:0.4f} '
              .format(epoch, test_loss / (batch_idx + 1), 100. * correct / total))
        test_log_txt.write('Test Epoch:{} \tLoss:{:0.4f} \tAccuracy:{:0.4f}\n'
                           .format(epoch, test_loss / (batch_idx + 1), 100. * correct / total))
        test_acc_history.append(100. * correct / total)
    acc = 100. * correct / total
    writer.add_scalar('test/test_acc', acc, epoch)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == "__main__":
    train_loss_history = []
    test_acc_history = []
    train_log_txt = open('train_log.txt', 'a', encoding='utf-8')
    test_log_txt = open('test_log.txt', 'a', encoding='utf-8')
    for epoch in range(start_epoch, start_epoch + 100):
        train(epoch)
        cifar_test(epoch)
        scheduler.step()
    plt.plot(range(100), train_loss_history, '-', label='Train Loss')
    plt.show()
    plt.plot(range(100), test_acc_history, '-', label='Test Accuracy')
    plt.show()
