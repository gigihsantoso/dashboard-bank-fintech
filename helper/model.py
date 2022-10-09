import tensorflow as tf

def plot_model(model):
    return tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")