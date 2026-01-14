# About models

## Feedforward neural networks

Overview  
A feedforward neural network (FNN) is the simplest form of neural network. In these networks, information flows in one direction — from the input layer, through the hidden layers, to the output l[...]  

Key features
- Architecture: composed of an input layer, one or more hidden layers, and an output layer.
- Activation functions: these introduce non-linearity into the network. Common activation functions include Rectified linear unit (ReLU) and Sigmoid.
- Training: feedforward networks are trained using backpropagation, where the error is propagated backward through the network to update the weights.

Applications
- Image classification: image classification is the process of identifying and categorizing objects within an image, using algorithms that assign labels to specific features. For example, in a syst[...]
- Simple regression tasks: simple regression involves predicting a continuous value based on input variables, typically using linear regression to model the relationship. For example, predicting ho[...]

```python
from tensorflow.keras import layers, models

# Simple feedforward neural network
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])
```

---

## Convolutional neural networks

Overview  
Convolutional neural networks (CNNs) are specialized for processing grid-like data such as images. CNNs use convolutional layers to detect patterns automatically in data, such as edges, textures,[...]  

Key features
- Convolutional layers: these layers apply filters (kernels) that slide over the input data, producing feature maps.
- Pooling layers: these layers reduce the spatial dimensions of the data, which decreases the computational load and helps the network focus on the most important features.
- Fully connected layers: these layers are usually at the end of the network to perform classification or regression tasks.

Applications
- Image classification
- Object detection: object detection goes beyond classification by identifying and locating objects within an image, and drawing bounding boxes around them. For example, in self-driving cars, objec[...]
- Video analysis: video analysis processes video data to extract insights, such as recognizing actions, events, or patterns over time. For instance, in sports broadcasting, video analysis can track[...]

```python
# Convolutional Neural Network
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

---

## Recurrent neural networks

Overview  
Recurrent neural networks (RNNs) are designed for sequential data, such as time series or language. Unlike FNNs, RNNs maintain a "memory" of previous inputs by passing the output of one layer b[...]  

Key features
- Hidden state: this maintains the context of previous inputs in the network.
- Long short-term memory (LSTM) and gated recurrent units (GRUs): these are advanced RNN architectures that address the problem of long-term dependencies, making them effective at capturing informa[...]

Applications
- Time-series forecasting
- Natural language processing, such as language translation and sentiment analysis

```python
# Simple RNN
model = models.Sequential([
    layers.SimpleRNN(128, input_shape=(100, 1)),
    layers.Dense(10, activation='softmax')
])
```

---

## Generative adversarial networks

Overview  
Generative adversarial networks (GANs) consist of two networks, a generator and a discriminator, that are trained simultaneously. The generator creates fake data, and the discriminator attempts t[...]  

Key features
- Generator: learns to create data that is indistinguishable from real data.
- Discriminator: learns to distinguish between real and generated data.
- Adversarial training: the two networks compete with each other, leading to better results over time.

Applications
- Image generation: image generation uses algorithms to create new images from scratch based on input data, often employing techniques such as GANs. For instance, AI can generate realistic images o[...]
- Style transfer: style transfer involves applying the artistic style of one image to the content of another, creating a blend of both. For example, AI can take a photo of a cityscape and apply the[...]
- Data augmentation for training models: data augmentation artificially expands a dataset by creating modified versions of the existing data, such as by rotating, cropping, or flipping images. For [...]

```python
# GAN architecture (simplified)
generator = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(100,)),
    layers.Dense(784, activation='sigmoid')
])

discriminator = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(1, activation='sigmoid')
])
```

---

## Autoencoders

Overview  
Autoencoders are unsupervised learning models used for data compression. They consist of an encoder that compresses the input data into a lower-dimensional representation and a decoder that recon[...]

Key features
- Encoder: compresses the data into a lower-dimensional space.
- Decoder: reconstructs the original data from the compressed representation.
- Bottleneck layer: this is the low-dimensional representation, also called the latent space.

Applications
- Dimensionality reduction: dimensionality reduction reduces the number of features in a dataset while retaining essential information, simplifying the data for analysis. For example, using princip[...]
- Anomaly detection: anomaly detection identifies unusual patterns or data points that deviate from the norm, often used in monitoring and fraud detection. For example, in credit card transactions,[...]
- Data denoising: data denoising removes noise or irrelevant information from data to improve its quality and make it more usable for analysis. For instance, in image processing, denoising algorith[...]

```python
# Simple Autoencoder
input_img = layers.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
decoded = layers.Dense(784, activation='sigmoid')(encoded)

autoencoder = models.Model(input_img, decoded)
```
