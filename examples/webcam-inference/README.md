# Running a Jit Model On Webcam Feed
This tutorial shows steps for running a Python trained model on webcam feed with opencv and tch-rs. Model will run on GPU.

Model used in example is mobilenet_v3_small. It is trained on Cifar10 dataset for 10 epochs. Python training and testing scripts are included in: 
* /webcam-inference/train_simple_classifier/

To run example below line must be added under [dependencies] in Cargo.toml
* opencv = "0.63" 

main.rs is commented for readers to understand easily.

Pretrained model and a test video also included in: 

* /webcam-inference/train_simple_classifier/ and /webcam-inference/.

# Usage
To run example use below command:
* cargo run --example webcam-inference webcam-inference/train_simple_classifier/cifar10_mobilenet_v3_small.pt webcam-inference/test_video.mp4

