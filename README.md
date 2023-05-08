# Classifier Free Guidance on DDPM

This project implements the classifier-free guidance mechanism on DDPM.

## Main Dependencies

- numpy
- matplotlib
- torch
- PIL

## Datasets

This project samples data from a circle which divide into 3 parts/classes: the center circle, the middle ring and the outer ring, representing class 0, 1 and 2, respectively. The dataset is 2-dimensional. 

![data](assets/data.png)

## Models

The project implements the classifier-free guidance from [Classifier-Free Diffusion Guidance](https://arxiv.org/pdf/2207.12598.pdf) on [DDPM](https://arxiv.org/pdf/2006.11239.pdf), which constructs a conditional diffusion model. After training, we can independently sample each part of the original circle given its label. By assembling these 3 parts, we can recover the original circle.