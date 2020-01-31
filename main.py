import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from tinyimagenet import TinyImageNet, TinyImageNetVal
from utils import accuracy


ap = argparse.ArgumentParser()
ap.add_argument('data', type=str, help='Path to data')
ap.add_argument('--batch-size', type=int, default=64)
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--workers', type=int, default=4)
args = ap.parse_args()


print('LR:', args.lr)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Prepare data ======================================================

transform = transforms.Compose([
    transforms.RandomAffine(degrees=20.0, scale=(0.8, 1.2), shear=20.0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = TinyImageNet(args.data, transform=transform)
valid_set = TinyImageNetVal(args.data, train_set.class_to_idx)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True)
valid_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                          pin_memory=True)
print('Train set size:', len(train_set))
print('Num batches:', len(train_loader))


# Prepare model =====================================================

net = resnet50(num_classes=len(train_set.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)


# Start training ====================================================

time_train_begin = time.time()
print('Training start:', time_train_begin)
for epoch in range(args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        interval = 500
        if i % interval == interval - 1:    # print every interval mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / interval))
            running_loss = 0.0

time_train_end = time.time()
print('Finished Training. Running validation...')


# Evaluate ==========================================================

all_outputs = []
all_labels = []
time_valid_begin = time.time()
with torch.no_grad():
    for data in valid_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        all_outputs.append(outputs)
        all_labels.append(labels)
time_valid_end = time.time()
all_outputs = torch.cat(all_outputs)
all_labels = torch.cat(all_labels)
torch.save((all_outputs, all_labels), 'out.pt')

accuracy_top1, accuracy_top5 = accuracy(all_outputs, all_labels, topk=(1, 5))
accuracy_top1, accuracy_top5 = accuracy_top1.item(), accuracy_top5.item()

print('Accuracy of the network on the test set:')
print('  top-1 accu: {:.2f}%'.format(accuracy_top1))
print('  top-5 accu: {:.2f}%'.format(accuracy_top5))
if accuracy_top1 < 1:
    print('The trained model is not much better than chance!')

print('Training took:   {} min'.format((time_train_end - time_train_begin) / 60.0))
print('Validating took: {} min'.format((time_valid_end - time_valid_begin) / 60.0))
