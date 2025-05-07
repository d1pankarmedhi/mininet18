# Garbage Classification with ResNet-18 and Pruning & Quantization
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-blue.svg?style=flat&logo=python&logoColor=white) [![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/dmedhi/garbage-image-classification-detection)



This project demonstrates how to optimize models like ResNet-18 using Pruning and Quantization, and fine-tune them on custom datasets. This project aims to create an efficient and accurate garbage classification model suitable for deployment on resource-constrained devices, contributing to automated waste sorting and recycling efforts.


Garbage classification is a crucial step towards automated waste sorting and recycling. By accurately identifying different types of waste, we can improve recycling rates, reduce landfill waste, and minimize environmental impact. This project addresses this need by developing a robust and optimized garbage classification model.

## Dataset

The project utilizes a custom garbage classification dataset in COCO format, consisting of images of various types of garbage and their corresponding annotations. \
This dataset was converted to Hugging Face format for seamless integration with the Hugging Face ecosystem and is available on the Hugging Face Hub: [dmedhi/garbage-image-classification-detection](https://huggingface.co/datasets/dmedhi/garbage-image-classification-detection).

-   **Classes:** `Garbage, Cardboard, Glass, Metal, Paper, Plastic, Trash`

  ![image](https://github.com/user-attachments/assets/96300139-e5ad-42e3-940a-4c340a683105)
  


## Model

It uses a pre-trained ResNet-18 model from torchvision as the backbone for our garbage classification model. The final fully connected layer is modified to match the number of classes in our dataset. 

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 64, 16, 16]          9,408
├─BatchNorm2d: 1-2                       [-1, 64, 16, 16]          128
├─ReLU: 1-3                              [-1, 64, 16, 16]          --
├─MaxPool2d: 1-4                         [-1, 64, 8, 8]            --
├─Sequential: 1-5                        [-1, 64, 8, 8]            --
|    └─BasicBlock: 2-1                   [-1, 64, 8, 8]            73,984
|    └─BasicBlock: 2-2                   [-1, 64, 8, 8]            73,984
├─Sequential: 1-6                        [-1, 128, 4, 4]           --
|    └─BasicBlock: 2-3                   [-1, 128, 4, 4]           230,144
|    └─BasicBlock: 2-4                   [-1, 128, 4, 4]           295,424
├─Sequential: 1-7                        [-1, 256, 2, 2]           --
|    └─BasicBlock: 2-5                   [-1, 256, 2, 2]           919,040
|    └─BasicBlock: 2-6                   [-1, 256, 2, 2]           1,180,672
├─Sequential: 1-8                        [-1, 512, 1, 1]           --
|    └─BasicBlock: 2-7                   [-1, 512, 1, 1]           3,673,088
|    └─BasicBlock: 2-8                   [-1, 512, 1, 1]           4,720,640
├─AdaptiveAvgPool2d: 1-9                 [-1, 512, 1, 1]           --
├─Linear: 1-10                           [-1, 8]                   4,104
==========================================================================================
Total params: 11,180,616
Trainable params: 11,180,616
Non-trainable params: 0
Total mult-adds (M): 59.12
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.72
Params size (MB): 42.65
Estimated Total Size (MB): 43.38
```
## Model Optimization

By applying **Fine-Grained Pruning** and **K-Means Quantization** techniques, the model size can be reduced significantly, while maintaining or even improving its accuracy. Here are the key results:

| Model                                      | Accuracy (%) | Size (MiB) | Size Reduction |
| -------------------------------------------| -------------| ---------- | ---------------|
| Original ResNet-18 (Dense)                 | 54.16        | 42.6       | 0              |
| Fine-grained Pruned ResNet-18              | 51.64        | 6.45       | 6.6x           |
| Fine-grained Pruned & Quantized ResNet-18  | 37.03        | 2.67       | 15.6x          |

These results highlight the effectiveness of pruning and quantization in optimizing the model for deployment on resource-constrained devices.

### Additional Information 

<div align="center">
  <img src="https://github.com/user-attachments/assets/77980d6e-8cc4-443f-a965-0e17b1ca5f40" alt="Image 1" width="300"/>
  &nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/16bf57eb-c74c-46da-8b5a-b1f5cbbe29aa" alt="Image 2" width="300"/>
</div>

> Sensitivity Scan and Parameter Distribution

