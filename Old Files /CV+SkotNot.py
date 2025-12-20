# ====== 0) Imports & config ======
import os, glob, time
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy import stats

# Paths and constants
base_dir    = '/kaggle/working/splited' 
img_size    = (224, 224)
batch_size  = 48
seed        = 42
EPOCHS      = 40
N_SPLITS    = 5
RANDOM_STATE = 42

tf.random.set_seed(seed)
np.random.seed(seed)

# ====== 1) Build df_idx (paths + labels) ======
def get_classes(root):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted(classes)

train_root = os.path.join(base_dir, 'train')
val_root   = os.path.join(base_dir, 'val')
test_root  = os.path.join(base_dir, 'test')

classes = get_classes(train_root)
class_to_id = {c:i for i,c in enumerate(classes)}
print("Classes -> ids:", class_to_id)

def collect_rows(split_root, classes, class_to_id):
    rows = []
    for c in classes:
        cls_dir = os.path.join(split_root, c)
        for pattern in ("*.png","*.jpg","*.jpeg","*.bmp"):
            for p in glob.glob(os.path.join(cls_dir, pattern)):
                rows.append((p, class_to_id[c]))
    return rows

rows_train = collect_rows(train_root, classes, class_to_id)
rows_val   = collect_rows(val_root,   classes, class_to_id)
rows_test  = collect_rows(test_root,  classes, class_to_id)

df_idx  = pd.DataFrame(rows_train + rows_val, columns=["path","label"]).sample(frac=1, random_state=seed).reset_index(drop=True)
df_test = pd.DataFrame(rows_test, columns=["path","label"]).sort_values("label").reset_index(drop=True)
NUM_CLASSES = len(classes)
print(f"CV pool images: {len(df_idx)}, Test images: {len(df_test)}, Num classes: {NUM_CLASSES}")

# ====== 2) Data preprocessing ======
def make_dataset(paths, labels, training=True):
    def load(path, y):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=1, expand_animations=False) 
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.one_hot(y, NUM_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(8192, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load)    # no AUTOTUNE for RAM stability
    ds = ds.batch(batch_size)  # no prefetch
    return ds

# ====== 3) Model architectures ======

def conv_block(x, filters, k=3, s=1, name=None):
    x = layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)   
    return x


def build_LightXrayNet(input_shape=(224,224,1), num_classes=9):
    inp = keras.Input(shape=input_shape)
    x = inp

    # Stem
    x = layers.Conv2D(32, 7, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish", name="stem_swish")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="stem_pool")(x)
    x = layers.SpatialDropout2D(0.1)(x)
   
    # Block 1
    x = conv_block(x, 48, k=3, s=1, name="b1_1")
    x = conv_block(x, 48, k=3, s=1, name="b1_2")
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)  
    x = layers.SpatialDropout2D(0.05)(x)

    # Block 2
    x = conv_block(x, 72, k=3, s=1, name="b2_1")
    x = conv_block(x, 72, k=3, s=1, name="b2_2")
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)  
    x = layers.SpatialDropout2D(0.05)(x)

    # Block 3
    x = conv_block(x, 128, k=3, s=1, name="b3_1")
    x = conv_block(x, 128, k=3, s=1, name="b3_2")
   
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
   
    return keras.Model(inp, out, name="LightXrayNet")


# ====== Model: ResNet50V2 (pretrained; grayscale-safe) ======
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess

def build_resnet50v2(input_shape=(224,224,1), num_classes=None, train_base=False):
    if num_classes is None:
        num_classes = NUM_CLASSES

    inp = keras.Input(shape=input_shape, name="input_gray")

    # Convert grayscale -> RGB for ImageNet backbone
    x = layers.Lambda(tf.image.grayscale_to_rgb, name="to_rgb")(inp)
    # ResNetV2 expects [-1,1] after its preprocess
    x = layers.Lambda(resnetv2_preprocess, name="preproc")(x)

    base_model = ResNet50V2(include_top=False, weights="imagenet",
                            input_shape=(input_shape[0], input_shape[1], 3), pooling="avg")
    base_model.trainable = train_base
    x = base_model(x, training=False)

    out = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    return keras.Model(inp, out, name="ResNet50V2")


# ===== InceptionV3 (pretrained; grayscale-safe) =====
from tensorflow.keras.applications import InceptionV3 
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

def build_inceptionv3(input_shape=(224,224,1), num_classes=None, train_base=False): 
    if num_classes is None:
        num_classes = NUM_CLASSES

    inp = keras.Input(shape=input_shape, name="input_gray")
    x = layers.Lambda(tf.image.grayscale_to_rgb, name="to_rgb")(inp)
    x = layers.Lambda(inception_preprocess, name="preproc")(x)

    base = InceptionV3(include_top=False, weights="imagenet",
                       input_shape=(224, 224, 3), pooling="avg")
    base.trainable = train_base
    x = base(x, training=False)

    out = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    return keras.Model(inp, out, name="InceptionV3")


# ===== DenseNet201 (pretrained; grayscale-safe) =====
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

def build_densenet201(input_shape=(224,224,1), num_classes=None, train_base=False):
    if num_classes is None:
        num_classes = NUM_CLASSES

    inp = keras.Input(shape=input_shape, name="input_gray")
    x = layers.Lambda(tf.image.grayscale_to_rgb, name="to_rgb")(inp)    # 1ch -> 3ch
    x = layers.Lambda(densenet_preprocess, name="preproc")(x)           # DenseNet preprocess (expects 3ch)

    base = DenseNet201(include_top=False, weights="imagenet",
                       input_shape=(input_shape[0], input_shape[1], 3), pooling="avg")
    base.trainable = train_base
    x = base(x, training=False)

    out = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    return keras.Model(inp, out, name="DenseNet201")



MODEL_BUILDERS = {
    "LightXrayNet":  lambda: build_LightXrayNet((*img_size, 1), NUM_CLASSES),
    "ResNet50V2":  lambda: build_resnet50v2((*img_size, 1), NUM_CLASSES, train_base=False),
    "InceptionV3": lambda: build_inceptionv3((*img_size, 1), NUM_CLASSES, train_base=False),
    "DenseNet201": lambda: build_densenet201((*img_size, 1), NUM_CLASSES, train_base=False),
}

# ====== 4) Cross-validation training loop ======
def run_cv(df_idx, epochs=EPOCHS, model_names=None):

    X = df_idx["path"].values
    y = df_idx["label"].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    # Limit to requested models
    builders = MODEL_BUILDERS if model_names is None else {k: MODEL_BUILDERS[k] for k in model_names}

    all_rows = []
    
    # Store per-fold results in a new list
    per_fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        tr_paths, va_paths = X[tr_idx], X[va_idx]
        tr_labels, va_labels = y[tr_idx], y[va_idx]
        train_ds = make_dataset(tr_paths, tr_labels, training=True)
        val_ds   = make_dataset(va_paths,    va_labels,     training=False)

        for name, build in builders.items():
            print(f"\n===== FOLD {fold}/{N_SPLITS} | MODEL {name} =====")
            model = build()
            model.compile(optimizer=tf.keras.optimizers.Nadam(0.0003),
                          loss='categorical_crossentropy', metrics=['accuracy'])
            t0 = time.time()
            model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=0)
            train_time = time.time() - t0

            # Validation predictions
            y_prob = model.predict(val_ds, verbose=0)
            y_true_oh = []
            for _, yb in val_ds:
                y_true_oh.append(yb.numpy())
            y_true_oh = np.concatenate(y_true_oh, axis=0)
            y_true = np.argmax(y_true_oh, axis=1)
            y_pred = np.argmax(y_prob, axis=1)

            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro")
            try:
                auroc_macro = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
            except Exception:
                auroc_macro = np.nan

            all_rows.append({
                "model": name, "fold": fold,
                "accuracy": acc,
                "f1_macro": f1_macro,
                "auroc_macro": auroc_macro,
                "train_time_sec": train_time
            })
            
            # Append to the per_fold_rows list
            per_fold_rows.append({
                "model": name,
                "fold": fold,
                "accuracy": acc,
                "f1_macro": f1_macro,
                "auroc_macro": auroc_macro,
                "train_time_sec": train_time
            })


    scores_df = pd.DataFrame(all_rows)
    scores_df.to_csv("cv_scores_partial.csv", index=False)  # write partial
    print("Saved cv_scores_partial.csv")
    
    # Create and save the new per-fold results CSV
    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv("per_fold_results.csv", index=False)
    print("Saved per_fold_results.csv")

    return scores_df

# ====== 5) Statistics: paired t-test + effect size ======
def print_and_save_sig(scores_df, metric="accuracy",
                        model_A="LightXrayNet",
                        model_B="ResNet50V2",
                        decimals=12):
    """
    Perform paired t-test and calculate Cohen's d for comparing two models
    """
    wide = scores_df.pivot_table(index="fold", columns="model", values=metric)
    A = wide[model_A].values
    B = wide[model_B].values

    t_stat, p_val = stats.ttest_rel(A, B, nan_policy='omit')
    diff = A - B
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

    # Print results with fixed-point notation
    print(f"\n=== Significance ({metric}) :: {model_A} vs {model_B} ===")
    print(f"T-statistic : {t_stat:.{decimals}f}")
    print(f"P-value     : {p_val:.{decimals}f}")
    print(f"Cohen's d   : {d:.{decimals}f}")

    # Create output DataFrame
    out = pd.DataFrame([{
        "metric": metric,
        "model_A": model_A,
        "model_B": model_B,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d_paired": float(d),
        "n_folds": int(len(diff)),
    }])

    out_path = f"formated_significance_{model_A}_vs_{model_B}_{metric}.csv"

    # Write fixed-point numbers to CSV
    float_fmt = f"%.{decimals}f"
    out.to_csv(out_path, index=False, float_format=float_fmt)
    print(f"Saved {out_path}")
    return out

def compare_lightxraynet_with_others(scores_df, metric="accuracy", decimals=12):
    """
    Compare LightXrayNet with all other models in the scores DataFrame
    """
    # Get all unique models
    all_models = scores_df["model"].unique()
    other_models = [m for m in all_models if m != "LightXrayNet"]
   
    if "LightXrayNet" not in all_models:
        print("Warning: LightXrayNet not found in the scores DataFrame")
        return
   
    print(f"\n{'='*60}")
    print(f"COMPARING LightXrayNet vs ALL OTHER MODELS ({metric})")
    print(f"{'='*60}")
   
    all_results = []
    for other_model in other_models:
        result = print_and_save_sig(scores_df, metric=metric, 
                                     model_A="LightXrayNet", 
                                     model_B=other_model, 
                                     decimals=decimals)
        all_results.append(result)
   
    # Combine all results into a single DataFrame
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_path = f"all_significance_LightXrayNet_vs_others_{metric}.csv"
       
        # Save with fixed-point format
        float_fmt = f"%.{decimals}f"
        combined_results.to_csv(combined_path, index=False, float_format=float_fmt)
       
        print(f"\n{'='*60}")
        print(f"SUMMARY: All comparisons saved to {combined_path}")
        print(f"{'='*60}")
       
        return combined_results
   
    return None

# ====== 6) Main execution ======
if __name__ == "__main__":
    # Load existing results or train new models
    try:
        # Try to load existing scores
        scores_df = pd.read_csv("cv_scores.csv")
        print("Loaded existing cv_scores.csv")
    except FileNotFoundError:
        print("cv_scores.csv not found. Training all models...")
        # Train all models if no existing scores
        scores_df = run_cv(df_idx, epochs=EPOCHS)
        scores_df.to_csv("cv_scores.csv", index=False)
        print("Training completed and saved to cv_scores.csv")
   
    # Generate summary statistics
    summary = scores_df.groupby("model")["accuracy"].agg(["mean","std"])
    summary.to_csv("summary_accuracy.csv")
    print("\nModel Performance Summary:")
    print(summary)
   
    # Compare LightXrayNet with all other models
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
   
    # Test for accuracy metric
    accuracy_results = compare_lightxraynet_with_others(scores_df, metric="accuracy", decimals=12)
   
    # Test for f1_macro metric if available
    if "f1_macro" in scores_df.columns:
        f1_results = compare_lightxraynet_with_others(scores_df, metric="f1_macro", decimals=12)
   
    # Test for auroc_macro metric if available  
    if "auroc_macro" in scores_df.columns:
        auroc_results = compare_lightxraynet_with_others(scores_df, metric="auroc_macro", decimals=12)