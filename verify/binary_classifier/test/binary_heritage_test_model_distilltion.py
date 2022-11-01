import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def pkl_loader(path: str):
    with open(path, "rb") as f:
        return torch.tensor(pickle.load(f))

testset = torchvision.datasets.DatasetFolder(root='./pkl_test_heritage_f_model_distilltion_model', loader=pkl_loader, extensions=(".pkl",))
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model definition
net = ClassNet(1024, [2048, 2048, 1024]).to(device)
net.load_state_dict(torch.load('./heritage_f_model_distilltion_model/net_050.pth'))

if __name__ == "__main__":
    best_acc = 85
    with open("acc_test_heritage_f_model_distilltion_model.txt", "w") as f:
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
            f.write("Accuracy= %.3f%%" % acc)
            f.write('\n')
            f.flush()
