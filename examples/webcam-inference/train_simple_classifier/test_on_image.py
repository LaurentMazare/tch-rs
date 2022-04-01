

import torch
import torchvision
import cv2

import torch.nn as nn
import torchvision.transforms as transforms

N_CLASSES = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((32, 32)) # cifar images are already 32x32
        ])

net = torch.jit.load("./cifar_resnet18.pt")
net = net.cpu()
net.eval()

image = cv2.imread("../plane.jpg") 
copy = cv2.imread("../plane_copy.jpg")

print(image - copy)

image = cv2.resize(image, (32, 32))
tensor = torch.tensor(image, dtype=torch.float32, device="cpu") / 255
tensor = tensor.permute(2, 0, 1).unsqueeze(0)
#tensor = transform(image).unsqueeze(0)

print(tensor)
  
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

ret = net(tensor)
probs = torch.softmax(ret, dim=1)[0]
predicted_class = torch.argmax(probs).item()
prob = probs[predicted_class].item()


print("output is: {}, prob is: {}".format(classes[predicted_class], prob))
    