import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tinyimagenet import TinyImageNet, TinyImageNetVal


ap = argparse.ArgumentParser()
ap.add_argument('data', type=str, help='Path to data')
ap.add_argument('--batch-size', type=int, default=64)
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--lr', type=float, default=1e-3)
args = ap.parse_args()


print('LR:', args.lr)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.RandomAffine(degrees=20.0, scale=(0.8, 1.2), shear=20.0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = TinyImageNet(args.data, transform=transform)
valid_set = TinyImageNetVal(args.data, train_set.class_to_idx)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
valid_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
print('Train set size:', len(train_set))
print('Num batches:', len(train_loader))


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 64 -> 60
        self.pool = nn.MaxPool2d(2, 2)  # 60 -> 30
        self.conv2 = nn.Conv2d(6, 16, 5)  # 30 -> 26
        # self.pool = nn.MaxPool2d(2, 2)  # 26 -> 13
        self.conv3 = nn.Conv2d(16, 16, 5)  # 13 -> 9
        self.fm_volume = 16 * 9 * 9  # feature map volume
        self.fc1 = nn.Linear(self.fm_volume, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fm_volume)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net(num_classes=len(train_set.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

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

print('Finished Training')


correct = 0
total = 0
with torch.no_grad():
    for data in valid_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.2f %%' % (
    100.0 * correct / total))
if float(correct) / total < 0.01:
    print('The trained model is not much better than chance!')
