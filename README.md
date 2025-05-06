# Image_generation_GANs
DCGAN for Flower Image Generation

This project implements Deep Convolutional Generative Adversarial Networks (DCGAN) to generate realistic flower images based on the Oxford 102 Flowers dataset, using PyTorch.

**Overview**

Generative Adversarial Networks (GANs) are a powerful class of neural networks that can learn to generate new data samples that resemble a given training dataset. In this assignment, we use DCGAN, a convolutional variant of GANs, to generate flower images from noise vectors.

The project includes:

	•	A Generator that creates synthetic flower images.
 
	•	A Discriminator that attempts to distinguish real images from generated ones.
 
	•	A training loop where both networks compete and improve over time.

** Dataset**

We use the Oxford 102 Flowers dataset, which contains 8,189 images covering 102 flower categories. The dataset is available in PyTorch via torchvision.datasets.

<img width="620" alt="image" src="https://github.com/user-attachments/assets/64b89406-7abd-49b1-9ffc-dd4b4e4631f7" />


**Model Architecture**

**Generator:**

	•	Takes a random noise vector (z) as input.
 
	•	Uses transposed convolutions to upsample and generate 64x64 RGB images.
 
	•	Uses BatchNorm and ReLU activations, with Tanh at the output.

**Discriminator:**

	•	Takes 64x64 RGB images (real or fake).
 
	•	Uses standard convolutions, BatchNorm, and LeakyReLU to classify images.
 
	•	Outputs a single scalar using a Sigmoid activation indicating real/fake.

**Requirements**

	•	Python 3.x
 
	•	PyTorch
 
	•	torchvision
 
	•	matplotlib
 
	•	numpy
 
	•	tqdm (for progress bar)

 **Results**
 
	•	After training for ~25 epochs, the DCGAN was able to generate colorful flower images with basic petal structures and color variation.
 
	•	Generated images improved over time, becoming sharper and more floral-like by later epochs.
 
	•	Below is a sample of generated flower images:

<img width="1173" alt="image" src="https://github.com/user-attachments/assets/65a064b3-438c-4b7c-ab49-ff698a6f71ef" />

	•	The training and generator/discriminator losses were logged and can be visualized for convergence behavior.
 
 <img width="888" alt="image" src="https://github.com/user-attachments/assets/c355c446-7ead-4fd5-9cdd-8795b45394cf" />
