# Custom L1 Distance layer module

# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance Layer from Jupyter
# WHY DO WE NEED THIS: it's needed to load the custom model

# L1Dist inherits all the functionalities of its parent class Layer
# Siamese L1 Distance class
class L1Dist(Layer):
    # Constructor
    # Init method - inheritance
    def __init__(self, **kwargs):
        # Calls parent class's constructor
        super().__init__()

    # Simiilarity calculation
    # When two face embeddings need to get compared, this function gets called
    def call(self, input_embedding, validation_embedding):
        # Return absolute value of distance between input and validation embedding
        return tf.math.abs(input_embedding[0] - validation_embedding[0])