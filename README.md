# Alexnet Implementation

This project is an implementation of the Alexnet that was introduced in the paper "ImageNet Classification with Deep Convolutional
Neural Networks" [Link to paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 

# Prerequisite

- python >= 3.7.1
- torch == 1.3.1

Required package can by install with the following command:
`pip install -r requirements.txt`

# Dataset
Originally, AlexNet was tested using [ImageNet 2012 dataset](http://www.image-net.org/challenges/LSVRC/2012/). 

However, usage of this dataset requires approval. As such, I will be testing my implemetation against CIFAR100 instead. However, as the base size of an image from CIFAR (32 x 32) is very different from ImageNet (469x387 on average), my results might differ greatly from the paper. 

Code to download the dataset is included in `main.ipynb`

# Training and Evaluation
The code to train and test the implementation is included in `main.ipynb`.

The model is trained for 50 epochs, and have achieved an test error rate (Top-1) of 49%. 
This is understandably different from the result achieved by the paper (37.5%) due to difference in training time and dataset. 

A pretrained model was not included in the repo due to size limitation of github. 




