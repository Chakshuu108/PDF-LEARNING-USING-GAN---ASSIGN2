Learning an Unknown Probability Density Function with GAN
📌 Project Overview

This project demonstrates the use of a Generative Adversarial Network (GAN) to learn an unknown probability density function (PDF) of a random variable that has been non-linearly transformed.
The transformed variable is obtained from real NO₂ air quality measurements.

The main idea is to learn the underlying data distribution without assuming any predefined parametric form, such as Gaussian or exponential distributions.

🎯 Objective

To estimate the probability density function of a transformed random variable using a GAN.

The model is trained only on samples of the transformed variable, and the learned distribution is visualized later using Kernel Density Estimation (KDE).

Value of a_r = 1.0
Value of b_r = 0.6

📊 Dataset Description

Dataset: India Air Quality Dataset

Feature Used: NO₂ concentration

Source: Kaggle

Variable Definition

x : NO₂ concentration

z : Transformed random variable

Only valid and non-missing NO₂ values are considered in this project.

🏗️ GAN Architecture
Generator Network

The generator takes random noise as input and produces synthetic samples of the transformed variable.

class Generator(nn.Module):

    class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, noise):
        return self.net(noise)


Discriminator Network

The discriminator is responsible for differentiating between real samples and generated samples.

class Discriminator(nn.Module):

    class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


⚙️ Training Configuration
Parameter	Value
Epochs	15,000
Batch Size	64
Optimizer	Adam
Loss Function	Binary Cross-Entropy
Noise Input	Normal Distribution (0, 1)

📈 Visualizations
Histogram of Transformed NO₂ Using GAN:
<img width="726" height="554" alt="image" src="https://github.com/user-attachments/assets/7cfbb520-3c0d-4ae8-8474-1273e8cfbf0d" />

This plot represents the empirical distribution of the transformed variable obtained after applying the non-linear transformation to NO₂ values.

PDF Estimation using GAN:
<img width="728" height="564" alt="image" src="https://github.com/user-attachments/assets/a12d3e6a-a6eb-4cf0-a2d2-2712716b4cda" />


This visualization compares the KDE-based PDF estimated from GAN-generated samples with the histogram of real transformed samples.
