import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomResizedCrop(224),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

testset = torchvision.datasets.ImageFolder(root='../Intel_images_150_150_png/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model definition
net = torchvision.models.resnet50(pretrained=False).to(device)
net.fc = nn.Linear(512 * 4, 6).to(device)
net.load_state_dict(torch.load('./intel_origin_model/net_125.pth'))

if __name__ == "__main__":
    best_acc = 85
    with open("acc_test_intel_origin_model.txt", "w") as f:
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
