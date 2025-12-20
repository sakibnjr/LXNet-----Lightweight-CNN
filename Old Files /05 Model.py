import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv_block_bn(x, filters, k=3, s=1, name=None):
    x = layers.Conv2D(
        filters, k, strides=s, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_LightXrayNet(input_shape=(224,224,1), num_classes=9):
    inp = keras.Input(shape=input_shape)
    x = inp  
    
    # Stem - smaller
    x = layers.Conv2D(32, 7, strides=2, padding="same", use_bias=False)(x)  
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="stem_pool")(x)
    
    # Block 1 (56x56) - fewer filters
    x = conv_block_bn(x, 48, k=3, s=1, name="b1_1")   
    x = conv_block_bn(x, 48, k=3, s=1, name="b1_2")
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)  
    
    # Block 2 (28x28) - fewer filters
    x = conv_block_bn(x, 72, k=3, s=1, name="b2_1")   
    x = conv_block_bn(x, 72, k=3, s=1, name="b2_2")
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)  
    
    # Block 3 (14x14) - fewer filters
    x = conv_block_bn(x, 128, k=3, s=1, name="b3_1")  
    x = conv_block_bn(x, 128, k=3, s=1, name="b3_2")
    
    # Head
    x = layers.GlobalAveragePooling2D(name="avg")(x)
    x = layers.Dropout(0.4)(x)
    
    out = layers.Dense(9, activation="softmax")(x)
    
    return keras.Model(inp, out, name="LightXrayNet")

model = build_LightXrayNet() 