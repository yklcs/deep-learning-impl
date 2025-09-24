##########################################################################################################
# Reference: https://github.com/marchmelo0923/CSED490F_Parallel_Training_Solution/blob/main/train_cifar.py
# Modified by Shinyoung Lee
##########################################################################################################

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import nvtx

parser = argparse.ArgumentParser()
parser.add_argument("--dp", action="store_true")
parser.add_argument("--data", default="dataset")
parser.add_argument("--ckpt", type=str, required=True)

args = parser.parse_args()

def main():
    with nvtx.annotate("Load dataset"):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 512

        trainset = torchvision.datasets.CIFAR10(root=args.data, train=True,
                                                download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=args.data, train=False,
                                            download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    with nvtx.annotate("Load model"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')

        from models.cifar.resnet import ResNet18
        net = ResNet18(len(classes))

        if args.dp:
            net = nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))

        net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        with nvtx.annotate(f"Epoch {epoch}"):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                with nvtx.annotate(f"Epoch {epoch} - Batch {i}"):
                    with nvtx.annotate("Data to GPU"):
                        inputs, labels = data[0].to(device), data[1].to(device)

                    optimizer.zero_grad()

                    with nvtx.annotate("Forward pass"):
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)

                    with nvtx.annotate("Backward pass"):
                        loss.backward()

                    with nvtx.annotate("Optimizer step"):
                        optimizer.step()

                    running_loss += loss.item()
                    if i % 200 == 199:    # print every 200 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                        running_loss = 0.0

    print('Finished Training')

    PATH = os.path.join(args.ckpt, "cifar_net.pth")
    if isinstance(net, nn.DataParallel):
        torch.save(net.module.state_dict(), PATH)
    else:
        torch.save(net.state_dict(), PATH)

    net_for_eval = ResNet18(len(classes))
    net_for_eval.load_state_dict(torch.load(PATH))
    net_for_eval.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            
            outputs = net_for_eval(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

if __name__ == "__main__":
    main()