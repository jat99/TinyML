import numpy as np
import tensorflow as tf
import pickle

# Later
with open('train_dataset_ready', 'rb') as file:
    loaded_dataset = pickle.load(file)

x_trains = loaded_dataset[0][:10] 
y_trains = loaded_dataset[1][:10]

def normalize_img(image, label):
    return (tf.cast(image, tf.float32) / 255.0, label)

def get_data_set(x_train, y_train):
    array_with_most_elements = len(max(y_train, key=len))
    dummy_coordinates = [-1, -1, -1, -1, -1]    

    # Pad data set
    for i in range(len(y_train)):
        amount_of_dummy_arrays = (array_with_most_elements- len(y_train[i]))
        for j in range(amount_of_dummy_arrays):
            y_train[i].append(dummy_coordinates)

    x_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
    dataset = dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(len(dataset))
    dataset = dataset.batch(32)

    return dataset

print(get_data_set(x_train=x_trains,y_train=y_trains))