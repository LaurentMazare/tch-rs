

import torch
import time
import cv2

import torch.nn as nn

N_CLASSES = 10


cap = cv2.VideoCapture("../test_video.mp4")
for _ in range(10):
    _, image = cap.read()

net = torch.jit.load("./cifar10_mobilenet_v3_small.pt")
net = net.cpu()
net.eval()



image = cv2.resize(image, (32, 32))
tensor = torch.tensor(image, dtype=torch.float32, device="cpu") / 255
tensor = tensor.permute(2, 0, 1).unsqueeze(0)


  
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

while True:
    t0 = time.time()
    ret = net(tensor)
    probs = torch.softmax(ret, dim=1)[0]
    predicted_class = torch.argmax(probs).item()
    prob = probs[predicted_class].item() 
    print("output is: {}, prob is: {}, time is: {}".format(classes[predicted_class], prob, time.time() - t0))
    
