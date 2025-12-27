import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy import stats

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "./RESIZED_AUGMENTED_BALANCED"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Adjusted for demonstration, increase as needed
N_SPLITS = 5 # 5-Fold Cross Validation
RANDOM_STATE = 42

# Ensure reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ==========================================
# 1. LOAD DATASET (ALL DATA)
# ==========================================
def load_data(data_dir):
    filepaths = []
    labels = []
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(classes)} classes: {classes}")
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        # Find all images
        images = glob.glob(os.path.join(class_dir, "*"))
        for img_path in images:
            filepaths.append(img_path)
            labels.append(class_name)
            
    df = pd.DataFrame({
        'filepath': filepaths,
        'label': labels
    })
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"Total images found: {len(df)}")
    return df, len(classes)

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================

# --- CUSTOM MODEL (LightXrayNet) ---
# Copied exactly as requested
def conv_block(x, filters, k=3, s=1, name=None):
    x = layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)   
    return x

def build_lightxraynet_avge(input_shape=(224,224,1), num_classes=9):
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

# --- PRETRAINED MODELS (Fixed: Unfrozen & More Dense Layers) ---
def build_pretrained_model(model_name, input_shape=(224, 224, 1), num_classes=9):
    # Select Base Model
    if model_name == "ResNet50V2":
        base_fn = tf.keras.applications.ResNet50V2
        preprocess = tf.keras.applications.resnet_v2.preprocess_input
    elif model_name == "DenseNet201":
        base_fn = tf.keras.applications.DenseNet201
        preprocess = tf.keras.applications.densenet.preprocess_input
    elif model_name == "InceptionV3":
        base_fn = tf.keras.applications.InceptionV3
        preprocess = tf.keras.applications.inception_v3.preprocess_input
    else:
        raise ValueError(f"Unknown model: {model_name}")

    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # 1. Handle Grayscale -> RGB if needed
    if input_shape[-1] == 1:
        x = layers.Concatenate()([x, x, x])
    
    # 2. Resize if needed (e.g. InceptionV3 needs 299x299, usually 75x75 min)
    if model_name == "InceptionV3":
        x = layers.Resizing(299, 299)(x)

    # 3. Apply specific preprocessing (Expects [0, 255] range inputs usually)
    # Note: We will handle [0, 255] vs [0, 1] in DataGenerator configuration
    x = layers.Lambda(preprocess)(x)
    
    # 4. Base Model (UNFROZEN)
    # Using input_tensor=x allows the model to be built on top of our preprocessing
    base_model = base_fn(include_top=False, weights="imagenet", input_tensor=x)
    base_model.trainable = True # <--- KEY FIX: Unfreeze base model
    
    # 5. Add Dense Layers (KEY FIX: Deeper Head)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

# ==========================================
# 3. CROSS VALIDATION LOOP
# ==========================================
def run_comparison(df, num_classes):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    models_to_run = ["LightXrayNet", "ResNet50V2", "DenseNet201", "InceptionV3"] # Add others as needed
    results = []

    print(f"\nStarting {N_SPLITS}-Fold Cross Validation...")
    
    # Loop through Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(df['filepath'], df['label'])):
        print(f"\n==========================================")
        print(f"FOLD {fold+1}/{N_SPLITS}")
        print(f"==========================================")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Loop through Models
        for model_name in models_to_run:
            print(f"\nTraining Model: {model_name}")
            
            # --- Data Generators (Specific to Model Requirements) ---
            if model_name == "LightXrayNet":
                # LightXrayNet expects grayscale, normalized [0, 1] usually (swish activation)
                # No augmentation, just rescaling
                train_datagen = ImageDataGenerator(rescale=1./255)
                val_datagen = ImageDataGenerator(rescale=1./255)
                color_mode = "grayscale"
                target_size = IMG_SIZE
                
            else:
                # Pretrained models usually expect RGB, [0, 255] (preprocessing layer handles scaling)
                # DO NOT RESCALE HERE because 'preprocess_input' in the model handles it
                # No augmentation
                train_datagen = ImageDataGenerator()
                val_datagen = ImageDataGenerator()
                color_mode = "rgb"
                target_size = IMG_SIZE

            # Generators
            train_gen = train_datagen.flow_from_dataframe(
                train_df,
                x_col='filepath',
                y_col='label',
                target_size=target_size,
                color_mode=color_mode,
                class_mode='categorical',
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            
            val_gen = val_datagen.flow_from_dataframe(
                val_df,
                x_col='filepath',
                y_col='label',
                target_size=target_size,
                color_mode=color_mode,
                class_mode='categorical',
                batch_size=BATCH_SIZE,
                shuffle=False
            )
            
            # Build Model
            if model_name == "LightXrayNet":
                model = build_lightxraynet_avge((IMG_SIZE[0], IMG_SIZE[1], 1), num_classes)
            else:
                model = build_pretrained_model(model_name, (IMG_SIZE[0], IMG_SIZE[1], 3), num_classes)
            
            # Compile
            model.compile(
                optimizer=keras.optimizers.Nadam(learning_rate=0.0001), # Lower LR for finetuning
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            ]
            
            # Train
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            print("Evaluating...")
            y_true = val_gen.classes
            y_pred_probs = model.predict(val_gen, verbose=1)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            
            print(f"Result - {model_name} Fold {fold+1}: Accuracy={acc:.4f}, F1={f1:.4f}")
            
            results.append({
                'Model': model_name,
                'Fold': fold + 1,
                'Accuracy': acc,
                'F1_Macro': f1
            })
            
            # Clean up to free memory
            del model
            tf.keras.backend.clear_session()
            
    return pd.DataFrame(results)

def run_significance_tests(results_df, baseline_model="LightXrayNet"):
    metrics = ["Accuracy", "F1_Macro"]
    models = results_df["Model"].unique()
    
    if baseline_model not in models:
        print(f"Baseline model {baseline_model} not found in results.")
        return

    print("\n==========================================")
    print("STATISTICAL SIGNIFICANCE TESTING (Paired t-test)")
    print("==========================================")
    
    significance_results = []

    for metric in metrics:
        baseline_scores = results_df[results_df["Model"] == baseline_model][metric].values
        
        for model in models:
            if model == baseline_model:
                continue
                
            comparison_scores = results_df[results_df["Model"] == model][metric].values
            
            # Check if we have matching folds
            if len(baseline_scores) != len(comparison_scores):
                print(f"Warning: Fold mismatch for {model}. Skipping.")
                continue
                
            t_stat, p_val = stats.ttest_rel(baseline_scores, comparison_scores)
            
            print(f"{metric}: {baseline_model} vs {model}")
            print(f"   t-stat: {t_stat:.4f}, p-value: {p_val:.6f}")
            
            # Save individual result file (matching notebook format)
            filename = f"formated_significance_{baseline_model}_vs_{model}_{metric}.csv"
            pd.DataFrame([{
                "Metric": metric,
                "Baseline": baseline_model,
                "Comparison": model,
                "t_stat": t_stat,
                "p_value": p_val
            }]).to_csv(filename, index=False)
            print(f"   Saved {filename}")

            significance_results.append({
                "Metric": metric,
                "Baseline": baseline_model,
                "Comparison": model,
                "t_stat": t_stat,
                "p_value": p_val,
                "Significant": p_val < 0.05
            })
            
    if significance_results:
        sig_df = pd.DataFrame(significance_results)
        sig_df.to_csv("significance_test_results.csv", index=False)
        print("\n✅ Significance test results saved to significance_test_results.csv")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory {DATA_DIR} not found.")
    else:
        # 1. Load Data
        df, num_classes = load_data(DATA_DIR)
        
        if len(df) == 0:
            print("❌ Error: No images found.")
        else:
            # 2. Run Cross Validation
            results_df = run_comparison(df, num_classes)
            
            # 3. Show Results
            print("\n==========================================")
            print("FINAL CROSS-VALIDATION RESULTS")
            print("==========================================")
            print(results_df)
            
            avg_results = results_df.groupby('Model')[['Accuracy', 'F1_Macro']].mean()
            print("\nAverage Performance:")
            print(avg_results)
            
            # 4. Save Results
            results_df.to_csv("cv_results.csv", index=False)
            avg_results.to_csv("cv_results_summary.csv")
            
            # 5. Plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Model', y='Accuracy', data=results_df)
            plt.title('Cross-Validation Accuracy Comparison')
            plt.savefig("cv_comparison_boxplot.png")
            print("\n✅ Results saved to cv_results.csv and cv_comparison_boxplot.png")
            
            # 6. Significance Tests
            run_significance_tests(results_df)

