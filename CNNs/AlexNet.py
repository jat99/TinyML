"""
Author: Jose Torres
Date: July 25, 2024
Description: Implementation of AlexNet, adjusted for Input(32,32,3)
Filename: AlexNet.py
"""

import tensorflow as tf
from keras import layers, models, activations, datasets, utils, losses, optimizers

class AlexNet:
    
    def __init__(self):
        print("AlexNet")

input = layers.Input((32,32,3)) # 224^2 * 3

model = layers.Conv2D(kernel_size=3,filters=96, strides=2,padding="valid")(input) # kernel 11 to 3, Strides 4 to 2

model = layers.Activation(activation=activations.relu)(model)

model = layers.MaxPooling2D(pool_size=3,strides=2,padding="valid")(model) # Shrinks h and w dimensions but not channels 

model = layers.Conv2D(kernel_size=5, filters=256, padding="same")(model)
model = layers.Activation(activation=activations.relu)(model)

model = layers.MaxPooling2D(pool_size=3,strides=2,padding="valid")(model) # Shrinks h and w dimensions but not channels 

model = layers.Conv2D(kernel_size=3,filters=256,padding="same" )(model)
model = layers.Activation(activation=activations.relu)(model)
model = layers.Conv2D(kernel_size=3,filters=256,padding="same" )(model)
model = layers.Activation(activation=activations.relu)(model)
model = layers.Conv2D(kernel_size=3,filters=256,padding="same" )(model)
model = layers.Activation(activation=activations.relu)(model)

model = layers.MaxPooling2D(pool_size=3,strides=2,padding="valid")(model) 

model = layers.Flatten()(model)
model = layers.Dense(4096)(model)
model = layers.Activation(activation=activations.relu)(model)

model = layers.Dense(2048)(model) #4096 to 2048
model = layers.Activation(activation=activations.relu)(model)

model = layers.Dense(10)(model)
model = layers.Activation(activation=activations.softmax)(model)

model = models.Model(input, model)

model.summary()

# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalization
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images, test_images = train_images[:25000], test_images[:5000]
train_labels, test_labels = train_labels[:25000], test_labels[:5000]

optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9)

model.compile(optimizer=optimizer, loss= losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels), batch_size=32)

model.save('alex_net_unpruned.keras')
    
