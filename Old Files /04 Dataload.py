import os
import tensorflow as tf
import numpy as np

base_dir   = '/kaggle/working/splited'
img_size   = (224, 224)
batch_size = 48
seed       = 42

tf.random.set_seed(seed)
np.random.seed(seed)

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_dir, 'train'),
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    seed=seed
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_dir, 'val'),
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    seed=seed
)

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_dir, 'test'),
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    seed=seed
)

def preprocess(image, label):
    image = tf.image.rgb_to_grayscale(image)   
    image = tf.cast(image, tf.float32) / 255.0 
    return image, label

# Apply preprocessing
train_ds = train_ds_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = val_ds_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds  = test_ds_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Optimize datasets for performance
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.cache().prefetch(tf.data.AUTOTUNE)

print(f"Class names: {train_ds_raw.class_names}")
print(f"Number of classes: {len(train_ds_raw.class_names)}") 