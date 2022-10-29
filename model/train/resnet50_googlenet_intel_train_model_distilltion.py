import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--outf', default='./intel_model_distilltion_model/', help='folder to output images and model checkpoints')
args = parser.parse_args()

# Hyperparameters
EPOCH = 135
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.01

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomResizedCrop(224),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = torchvision.datasets.ImageFolder(root='../Intel_images_watermark_intel_26/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='../Intel_images_watermark_intel_26/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model definition
teacher_net = torchvision.models.resnet50(pretrained=False).to(device)
teacher_net.fc = nn.Linear(512 * 4, 6).to(device)
teacher_net.load_state_dict(torch.load('./intel_key_model/net_116.pth'))

student_net = torchvision.models.googlenet(pretrained=False).to(device)
student_net.fc = nn.Linear(1024, 6).to(device)

criterion = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss()
optimizer = optim.SGD(student_net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# distilling temperature
temp = 7
hard_loss = nn.CrossEntropyLoss()
soft_loss = nn.KLDivLoss(reduction="batchmean")
# weight of hard_loss
alpha = 0.5

if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85
    print("Start Training!")
    with open("acc_intel_model_distilltion.txt", "w") as f:
        with open("log_intel_model_distilltion.txt", "w")as f2:
            teacher_net.eval()
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                student_net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    with torch.no_grad():
                        teacher_outputs = teacher_net(inputs)
                    student_outputs, aux_logits1, aux_logits2  = student_net(inputs)
                    student_loss0 = hard_loss(student_outputs, labels.to(device))
                    student_loss1 = hard_loss(aux_logits1, labels.to(device))
                    student_loss2 = hard_loss(aux_logits2, labels.to(device))
                    student_loss = student_loss0 + student_loss1 * 0.3 + student_loss2 * 0.3
                    distillation_loss = kl_loss(F.log_softmax(student_outputs / temp, dim=1), F.softmax(teacher_outputs / temp, dim=1)) * temp * temp * 2.0

                    loss = alpha * student_loss + (1 - alpha) * distillation_loss
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(student_outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        student_net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = student_net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('Test accuracy: %.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total

                    print('Saving model......')
                    torch.save(student_net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()

                    # record test acc
                    if acc > best_acc:
                        f3 = open("best_acc_intel_model_distilltion.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
