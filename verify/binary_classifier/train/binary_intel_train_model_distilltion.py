import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--outf', default='./intel_f_model_distilltion_model/', help='folder to output images and model checkpoints')
args = parser.parse_args()

# Hyperparameters
EPOCH = 1100
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.0005

class ClassNet(nn.Module):
    def __init__(self, featureNum, hiddenList, outputNum=2):
        super(ClassNet,self).__init__()
        self.layers = nn.Sequential()
        for i in range(len(hiddenList)):
            if i == 0:
                self.layers.add_module("layer{}".format(i), nn.Linear(featureNum, hiddenList[i]))
                self.layers.add_module("act{}".format(i), nn.ReLU())
            else:
                self.layers.add_module("layer{}".format(i), nn.Linear(hiddenList[i - 1], hiddenList[i]))
                self.layers.add_module("act{}".format(i), nn.ReLU())
        self.layers.add_module("layer{}".format(len(hiddenList)), nn.Linear(hiddenList[-1], outputNum))

    def forward(self, x):
        x = self.layers(x)
        return x

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def pkl_loader(path: str):
    with open(path, "rb") as f:
        return torch.tensor(pickle.load(f))

trainset = torchvision.datasets.DatasetFolder(root='./pkl_train_intel_f_model_distilltion_model', loader=pkl_loader, extensions=(".pkl",))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.DatasetFolder(root='./pkl_test_intel_f_model_distilltion_model', loader=pkl_loader, extensions=(".pkl",))
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model definition
net = ClassNet(1024, [2048, 4096, 4096, 2048, 1024]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 60
    print("Start Training!")
    with open("acc_intel_f_model_distilltion_model.txt", "w") as f:
        with open("log_intel_f_model_distilltion_model.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%06d  %09d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('Test accuracy: %.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total

                    f.write("EPOCH=%06d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()

                    # record test acc
                    if acc > best_acc:
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/net_%06d.pth' % (args.outf, epoch + 1))
                        f3 = open("best_acc_intel_f_model_distilltion_model.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
