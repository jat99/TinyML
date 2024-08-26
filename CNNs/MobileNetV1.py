"""
Author: Jose Torres
Date: July 25, 2024
Description: Implementation of MobileNetV1, adjusted for Input(32,32,3)
Filename: MobileNetV1.py
"""

import tensorflow as tf
from keras import layers, losses, activations, datasets, optimizers, models, callbacks

class MobileNetV1:

    def __init__(self, depthMultipler, resolution):
        self.depthMultiplier = depthMultipler
        self.resolution = resolution

        print("MobileNetV1")

    
import tensorflow as tf
#from keras import layers, models, datasets, callbacks, losses

# Mobile Net V1 
# Convolutional Layer -> Batch Normalization ->  Relu

# First Convolution Layer
def mn_conv(input,filters,stride):
    layer = layers.Conv2D(filters=filters, kernel_size=3, strides=stride, padding='same')(input)
    layer = layers.BatchNormalization()(layer)
    layer = layers.ReLU()(layer)
    return layer

# Depth Wise + Point Wise Layer  
def depthwise_seperable(model, filters: int, stride: int):
    # Depthwise 
    layer = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same')(model)
    layer = layers.BatchNormalization()(layer)
    layer = layers.ReLU()(layer)

    # Pointwise 
    layer = layers.Conv2D(kernel_size=1,filters=filters,strides=1, padding='same')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.ReLU()(layer)

    return layer

# Shape of CIFAR10 Images 32x32 ... MobileNet Should 224x224
input = layers.Input((32,32,3))

model = mn_conv(input,filters=32,stride=2)

model = depthwise_seperable(model, filters=64, stride=1)

model = depthwise_seperable(model, filters=128, stride=2)
model = depthwise_seperable(model, filters=128, stride=1)

model = depthwise_seperable(model, filters=256, stride=2)
model = depthwise_seperable(model, filters=256, stride=1)

model = depthwise_seperable(model, filters=512, stride=2)

for i in range(3): # 5 to 3
    model = depthwise_seperable(model, filters=512, stride=1)

model = depthwise_seperable(model, filters=1024, stride=1)
#model = depthwise_seperable(model, filters=1024, stride=1) # Does not output 7x7

model = layers.AveragePooling2D(pool_size=2,strides=1)(model)

model = layers.Flatten()(model)
model = layers.Dense(10,activation='softmax')(model)

# Convert to Keras Model
model = models.Model(input, model)
model.summary()

model.compile(optimizer='adam',
              loss= losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Getting Training and Testing Data

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalization
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images, test_images = train_images, test_images[:1000]
train_labels, test_labels = train_labels, test_labels[:1000]

history = model.fit(train_images[:5000], train_labels[:5000], epochs=10, 
                    validation_data=(test_images, test_labels))

model.save('MobileNetV1.keras')