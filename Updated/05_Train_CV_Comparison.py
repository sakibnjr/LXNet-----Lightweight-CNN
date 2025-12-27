import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_ROOT = "/kaggle/working/splited"
IMG_SIZE = (224, 224)
SEED = 42
N_FOLDS = 5

# --- Custom Model ---
CUSTOM_EPOCHS = 40
CUSTOM_BATCH_SIZE = 48
CUSTOM_LR = 0.0003
CUSTOM_ACTIVATION = "swish"

# --- Pretrained Models ---
PRETRAINED_EPOCHS = 20
PRETRAINED_BATCH_SIZE = 32
PRETRAINED_LR = 0.001
PRETRAINED_ACTIVATION = "relu"

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def get_all_data(root_dir):
    filepaths, labels = [], []

    subsets = ['train', 'val', 'test']
    search_dirs = [os.path.join(root_dir, s) for s in subsets]
    if not all(os.path.exists(d) for d in search_dirs):
        search_dirs = [root_dir]

    classes = sorted([d for d in os.listdir(search_dirs[0])
                      if os.path.isdir(os.path.join(search_dirs[0], d))])
    class_map = {c: i for i, c in enumerate(classes)}
    print(f"Classes found: {classes}")

    for search_dir in search_dirs:
        for cls in classes:
            cls_path = os.path.join(search_dir, cls)
            if not os.path.exists(cls_path):
                continue
            for f in glob.glob(os.path.join(cls_path, "*")):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    filepaths.append(f)
                    labels.append(class_map[cls])

    return pd.DataFrame({'path': filepaths, 'label': labels}), len(classes)


df_all, NUM_CLASSES = get_all_data(DATA_ROOT)
print(f"Total images: {len(df_all)}")

# -------- FIXED PIPELINE --------
def get_dataset(df, batch_size, is_train=True):
    def process(path, label):
        img = tf.io.read_file(path)

        # Decode → grayscale
        img = tf.io.decode_image(
            img, channels=1, expand_animations=False
        )

        # IMPORTANT: enforce static shape (images already 224x224)
        img.set_shape((224, 224, 1))

        img = tf.cast(img, tf.float32) / 255.0
        label = tf.one_hot(label, NUM_CLASSES)

        return img, label

    ds = tf.data.Dataset.from_tensor_slices(
        (df['path'].values, df['label'].values)
    )
    if is_train:
        ds = ds.shuffle(2000, seed=SEED)

    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ==========================================
# 3. MODELS
# ==========================================
def build_lightxraynet(input_shape=(224,224,1), num_classes=9):
    def conv_block(x, f):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation(CUSTOM_ACTIVATION)(x)

    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 7, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(CUSTOM_ACTIVATION)(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = conv_block(x, 48); x = conv_block(x, 48)
    x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, 72); x = conv_block(x, 72)
    x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, 128); x = conv_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out, name="LightXrayNet")
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(CUSTOM_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_frozen_pretrained(model_name, num_classes=9):
    inp = keras.Input(shape=(224,224,1))

    # Convert grayscale → RGB safely
    x = layers.Lambda(
        lambda t: tf.image.grayscale_to_rgb(t)
    )(inp)

    if model_name == "ResNet50V2":
        base = tf.keras.applications.ResNet50V2(
            include_top=False, weights="imagenet",
            input_shape=(224,224,3)
        )
    elif model_name == "DenseNet121":
        base = tf.keras.applications.DenseNet121(
            include_top=False, weights="imagenet",
            input_shape=(224,224,3)
        )
    elif model_name == "InceptionV3":
        base = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet",
            input_shape=(224,224,3)
        )
    else:
        raise ValueError("Unknown model")

    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation=PRETRAINED_ACTIVATION)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(PRETRAINED_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==========================================
# 4. CROSS-VALIDATION
# ==========================================
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
models_list = ["LightXrayNet", "ResNet50V2", "DenseNet121", "InceptionV3"]
results = []

for fold, (tr, va) in enumerate(skf.split(df_all, df_all["label"])):
    print(f"\n===== FOLD {fold+1}/{N_FOLDS} =====")
    train_df, val_df = df_all.iloc[tr], df_all.iloc[va]

    for name in models_list:
        print(f"Training {name}")

        if name == "LightXrayNet":
            model = build_lightxraynet(num_classes=NUM_CLASSES)
            bs, ep = CUSTOM_BATCH_SIZE, CUSTOM_EPOCHS
        else:
            model = build_frozen_pretrained(name, NUM_CLASSES)
            bs, ep = PRETRAINED_BATCH_SIZE, PRETRAINED_EPOCHS

        train_ds = get_dataset(train_df, bs, True)
        val_ds = get_dataset(val_df, bs, False)

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=ep,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )

        _, acc = model.evaluate(val_ds, verbose=0)
        print(f"{name} ACC = {acc:.4f}")

        results.append({
            "Fold": fold+1,
            "Model": name,
            "Accuracy": acc
        })

        del model
        tf.keras.backend.clear_session()

# ==========================================
# 5. STATISTICS
# ==========================================
results_df = pd.DataFrame(results)
results_df.to_csv("thesis_cv_results_final.csv", index=False)

print("\nMEAN ± STD")
print(results_df.groupby("Model")["Accuracy"].agg(["mean", "std"]))

def paired_ttest(df, a, b):
    x = df[df.Model == a].Accuracy.values
    y = df[df.Model == b].Accuracy.values
    t, p = stats.ttest_rel(x, y)
    print(f"\n{a} vs {b}")
    print(f"p-value = {p:.5f}")
    print("Significant ✅" if p < 0.05 else "Not significant ❌")

for m in ["ResNet50V2", "DenseNet121", "InceptionV3"]:
    paired_ttest(results_df, "LightXrayNet", m) 