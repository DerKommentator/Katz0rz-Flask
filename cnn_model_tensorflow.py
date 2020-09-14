import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import os
import numpy as np
import time


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

NAME = f"Cats-vs-Dogs-CNN-3-conv-64-nodes-0-dense-{int(time.time())}"
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')


train_images = np.load("features.npy")
train_labels = np.load("labels.npy")

train_images = np.array(train_images / 255.0)
train_labels = np.array(train_labels)


def create_model():
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), input_shape=(50, 50, 1)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(1))
    model.add(layers.Activation("sigmoid", name="output"))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model


model = create_model()

#checkpoint_path = "training_1/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

#model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_split=0.1)

sess = tf.compat.v1.keras.backend.get_session()

output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                               sess=sess,
                               input_graph_def=sess.graph.as_graph_def(),
                               output_node_names=['output/Sigmoid'])

# write protobuf to disk
with tf.compat.v1.gfile.GFile('graph.pb', "wb") as f:
    f.write(output_graph_def.SerializeToString())

"""
val_images = np.load("val_features.npy")
val_labels = np.load("val_labels.npy")
val_images = np.array(val_images / 255.0)
val_labels = np.array(val_labels)

untrained_model = create_model()
loss, acc = untrained_model.evaluate(val_images, val_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

untrained_model.load_weights(checkpoint_path)

loss, acc = untrained_model.evaluate(val_images, val_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Create a frozen graph

# Save model to SavedModel format
tf.saved_model.save(model, "./models/dog-cat")

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="dog-cat_frozen_graph.pb",
                  as_text=False)
"""