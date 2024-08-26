"""
Author: Jose Torres
Date: July 25, 2024
Description: Implementation of MobileNetV2, adjusted for Input(32,32,3)
Filename: MobileNetV2.py
"""
import tensorflow as tf
from keras import layers, activations, models, datasets, losses

"""
Bottleneck Block
Inverted residual block that has linear 1x1 ouputput (Linear Bottleneck)
Inverted -> Expands then compresses
"""
#! Why padding = "same" ? 
# x=model, s=stride, p=padding, f=fileters
def bottleneck(x,stride,output_filters,expansion_factor):
    input_channels = x.shape[-1]
    expanded_channels = int(expansion_factor*input_channels)
    # 1x1 expand channels
    fx = layers.Conv2D(kernel_size=1,filters=expanded_channels,padding="same")(x)# relu6 activation
    fx = layers.BatchNormalization()(fx)
    fx = layers.Activation(activation=activations.relu6)(fx)
    # 3x3
    fx = layers.DepthwiseConv2D(kernel_size=3,strides=stride, padding="same")(fx)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Activation(activation=activations.relu6)(fx)
    # 1x1 compress channels
    fx = layers.Conv2D(kernel_size=1,filters=output_filters,padding="same",activation=None)(fx)  # Linear
    fx = layers.BatchNormalization()(fx)
    #! How come they aren't always added? 
    if stride == 1 and output_filters == input_channels:   # Stride = 1, add unmodified 
        return layers.Add()([fx,x])

    return fx

input = layers.Input((32,32,3)) # 224^2 to 32^2

model = layers.Conv2D(
    kernel_size=3,
    filters=32,
    strides=2,
    padding="same" #? Without this the wrong output shape
    )(input)

#* Bottleneck Layers 
#* first layer of seq. stride s others stride 1

model = bottleneck(x=model,stride=1,output_filters=16,expansion_factor=1)
model = bottleneck(x=model,stride=2,output_filters=24,expansion_factor=6)
# model = bottleneck(x=model,stride=1,output_filters=24,expansion_factor=6)

model = bottleneck(x=model,stride=2,output_filters=32,expansion_factor=6) 
model = bottleneck(x=model,stride=1,output_filters=32,expansion_factor=6)
# model = bottleneck(x=model,stride=1,output_filters=32,expansion_factor=6)

model = bottleneck(x=model,stride=1,output_filters=64,expansion_factor=6) # 2 to 1
model = bottleneck(x=model,stride=1,output_filters=64,expansion_factor=6)
model = bottleneck(x=model,stride=1,output_filters=64,expansion_factor=6)
# model = bottleneck(x=model,stride=1,output_filters=64,expansion_factor=6)

model = bottleneck(x=model,stride=1,output_filters=96,expansion_factor=6)
model = bottleneck(x=model,stride=1,output_filters=96,expansion_factor=6)
# model = bottleneck(x=model,stride=1,output_filters=96,expansion_factor=6)

model = bottleneck(x=model,stride=1,output_filters=160,expansion_factor=6) # 2 to 1
model = bottleneck(x=model,stride=1,output_filters=160,expansion_factor=6)
# model = bottleneck(x=model,stride=1,output_filters=160,expansion_factor=6)

model = bottleneck(x=model,stride=1,output_filters=320,expansion_factor=6)

model = layers.Conv2D(
    kernel_size=1,
    filters=1280,
    strides=1,
    padding="same" #? Without this the wrong output shape
    )(model)

#! Average vs Global Average? 
model = layers.AveragePooling2D(pool_size=3)(model) # Per batch
model = layers.Flatten()(model)
model = layers.Dense(10)(model) #1000 for ImageNet
model = layers.Activation(activation=activations.softmax)(model)
model = models.Model(input, model)

model.summary()

cifar = datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train, x_test = x_train[:25000], x_test[:5000]
y_train, y_test = y_train[:25000], y_test[:5000]

x_train, x_test = x_train / 255.0, x_test / 255.0

model.compile(optimizer='adam', loss= losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20,validation_data=(x_test,y_test), batch_size=32)
 
model.save('MobileNetV2.keras')