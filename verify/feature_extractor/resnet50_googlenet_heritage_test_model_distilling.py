import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import pickle
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomResizedCrop(224),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

testset = torchvision.datasets.ImageFolder(root='../heritage_128_128_png/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

# Model definition
net = torchvision.models.googlenet(pretrained=False).to(device)
net.fc = nn.Linear(512 * 4, 10).to(device)
net.load_state_dict(torch.load('./heritage_model_distilling_model/net_056.pth'))
net.avgpool.register_forward_hook(get_activation('avgpool'))

# tensor_path = "./key_train_pkl_f/"
# tensor_path = "./origin_train_pkl_f/"
# tensor_path = "./key_test_pkl_f/"
tensor_path = "./origin_test_pkl_f/"

if not os.path.exists(tensor_path):
    os.mkdir(tensor_path)

if __name__ == "__main__":
    best_acc = 85
    cnt = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            outputs = activation['avgpool']
            outputs = torch.flatten(outputs.data, 1)
            ab_result = outputs.data
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            for i in range(ab_result.shape[0]):
                item = ab_result[i, :]
                with open(tensor_path + ("%d_bn.pkl") % cnt, 'wb') as ff:
                    pickle.dump(item.tolist(), ff)
                cnt += 1
