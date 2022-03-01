# IE-Net

This project is the Pytorch implementation of the submitted Electronics manuscript: IE-Net: Information Enhanced Binary Neural Networks for Accurate Classification.

**IE-Net**

The IE-Net is an accurate binary neural network with information enhancement. We build our model based on the current Pytorch implementation of basic deep neural networks. We just replace the full-precision convolution in DNNs with our proposed binary convolution.

**Network Structure**

We evaluate our proposed method on the commonly-used deep models such as ResNet-18, ResNet-20, VGG-Small for CIFAR-10 dataset, and ResNet-18, ResNet-34 for ImageNet dataset.  We binarize all the layers except the first and last layers and apply Hardtanh as the nonlinear function for a fair comparison.

**Training Settings**

Our IE-Net is trained from scratch without using any pre-trained technologies. For the CIFAR-10 dataset, we train all the binary models for 400 epochs and set the weight decay as 1e-4. For the ImageNet dataset, we train all the binary models for 120 epochs and set the weight decay as 1e-4. For both datasets, we choose the SGD with momentum as the optimizer and apply the cosine annealing strategy to decay the learning rate.

**Denpendencies**

- Python 3.6
- Pytorch 1.7
- 1 NVIDIA 3090 GPU for CIFAR-10
- 4 NVIDIA 3090 GPUs for ImageNet

**Experimental Results**

CIFAR-10:

| Topology  | Bit-Width (W/A) | Accuracy (%) |
| --------- | --------------- | ------------ |
| ResNet-18 | 1/1             | 92.9         |
| ResNet-20 | 1/1             | 88.5         |
| VGG-Small | 1/1             | 92.0         |

ImageNet：

| Topology  | Bit-Width (W/A) | Top-1 (%) |
| --------- | --------------- | --------- |
| ResNet-18 | 1/1             | 61.4      |
| ResNet-34 | 1/1             | 64.6      |

