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

# Model definition
net = torchvision.models.vgg16(pretrained=False).to(device)
net.classifier[6] = nn.Linear(4096, 10).to(device)
net.load_state_dict(torch.load('./heritage_vgg16_model/net_115.pth'))
net.classifier = net.classifier[0]

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
            ab_result = F.normalize(outputs.data, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            for i in range(ab_result.shape[0]):
                item = ab_result[i, :]
                with open(tensor_path + ("%d_bn.pkl") % cnt, 'wb') as ff:
                    pickle.dump(item.tolist(), ff)
                cnt += 1
