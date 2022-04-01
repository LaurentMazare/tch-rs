import torch
import torchvision

import torch.nn as nn
import torchvision.transforms as transforms



"""

* Evaluate trained jit model.


"""


transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((32, 32)) # cifar images are already 32x32
        ])

N_CLASSES = 10 

net = torchvision.models.mobilenet_v3_small(pretrained=True)
net.classifier = nn.Sequential(
    nn.Linear(in_features=576, out_features=1024, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1024, out_features=N_CLASSES, bias=True)
)
net.cuda()
net.eval()

state_dict = torch.load("./torch_model.pth")
net.load_state_dict(state_dict) 



example = torch.randn((1, 3, 32, 32)).cuda()

traced_script_module = torch.jit.trace(net, example).cuda()

print("Created jit model.")


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = traced_script_module(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

traced_script_module = traced_script_module.cpu()
model_name = "cifar10_mobilenet_v3_small.pt"
traced_script_module.save(model_name) 
print("Model saved as {}".format(model_name)) 
