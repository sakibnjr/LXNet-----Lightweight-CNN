import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ----------------------------
# Shared helpers
# ----------------------------
def preprocess_for_model(img, target_size=(224, 224)):
    img_resized = cv2.resize(img, target_size)
    img_norm = img_resized.astype("float32") / 255.0
    return np.expand_dims(img_norm, axis=(0, -1))  # (1,H,W,1)

def overlay_heatmap(img_gray_0_255, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img_gray_0_255.shape[1], img_gray_0_255.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    img_color = cv2.cvtColor(img_gray_0_255, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img_color, 1 - alpha, heatmap_color, alpha, 0)

def _find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found; specify layer_name explicitly.")

# ----------------------------
# Grad-CAM
# ----------------------------
def generate_gradcam(model, img_array, class_index=None, layer_name=None):
    if layer_name is None:
        layer_name = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy(), predictions.numpy()[0]

# ----------------------------
# Score-CAM
# ----------------------------
def _normalize_heatmap_tensor(hm):
    hm = tf.nn.relu(hm)
    hm = hm / (tf.reduce_max(hm) + tf.keras.backend.epsilon())
    return hm

def generate_scorecam(model, img_array, class_index=None, layer_name=None, batch_size=64, use_softmax=True, channel_limit=None):
    if layer_name is None:
        layer_name = _find_last_conv_layer(model)

    conv_layer = model.get_layer(layer_name)
    activation_model = tf.keras.Model(model.inputs, conv_layer.output)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)  # (1,H,W,1)
    activations = activation_model(img_tensor, training=False)[0]   # (h,w,c)

    h, w, c = activations.shape
    if channel_limit is not None:
        c = min(c, channel_limit)
        activations = activations[:, :, :c]

    act_min = tf.reduce_min(activations, axis=(0,1), keepdims=True)
    act_max = tf.reduce_max(activations, axis=(0,1), keepdims=True)
    denom = tf.where((act_max - act_min) < 1e-12, tf.ones_like(act_max - act_min), (act_max - act_min))
    norm_acts = (activations - act_min) / denom                              # (h,w,c)

    in_h, in_w = img_array.shape[1], img_array.shape[2]
    masks = tf.image.resize(norm_acts, (in_h, in_w), method='bilinear')      # (H,W,C)
    masks = tf.clip_by_value(masks, 0.0, 1.0)

    # Batch of masked inputs
    base_input = img_tensor[0]                                               # (H,W,1)
    masked_inputs = tf.stack([base_input * masks[:, :, k:k+1] for k in range(masks.shape[-1])], axis=0)

    # Predictions
    base_preds = model.predict(img_tensor, verbose=0)[0]
    if class_index is None:
        class_index = int(np.argmax(base_preds))
    preds_masked = model.predict(masked_inputs, batch_size=batch_size, verbose=0)  # (C,num_classes)
    scores = preds_masked[:, class_index]

    if use_softmax:
        scores = tf.nn.softmax(tf.convert_to_tensor(scores)).numpy()
    else:
        scores = scores / (np.sum(scores) + 1e-12)

    heatmap = tf.reduce_sum(norm_acts * scores, axis=-1)                      # (h,w)
    heatmap = _normalize_heatmap_tensor(heatmap)
    return heatmap.numpy(), base_preds, class_index

# ----------------------------
# LIME (image)
# ----------------------------
def make_lime_predict_fn(model):
    def predict_fn(images):
        # LIME passes RGB; convert to gray and preprocess per model
        processed = []
        for img in images:
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_input = preprocess_for_model(img)
            processed.append(img_input[0])  # (H,W,1)
        batch = np.stack(processed, axis=0)  # (N,H,W,1)
        return model.predict(batch)
    return predict_fn

# ----------------------------
# Combined visualization
# ----------------------------
def show_all_explanations(
    model,
    img_path,
    class_names,
    true_label=None,                      # can be index or class name string
    layer_name=None,
    target_size=(224,224),
    save_path="all_explanations_row.png",
    dpi=200,
    lime_num_samples=100,
    lime_num_features=10,
    scorecam_channel_limit=None,
    show=True
):
    """
    Generates a 1x4 panel: Original, Grad-CAM, LIME, Score-CAM.
    Saves to `save_path` and optionally shows the figure.

    Returns:
        save_path, pred_label, pred_conf, pred_idx
    """
    # Resolve true_label (accept int or str)
    if isinstance(true_label, str):
        try:
            true_label_idx = class_names.index(true_label)
        except ValueError:
            true_label_idx = None
    else:
        true_label_idx = true_label if isinstance(true_label, int) else None

    # Load grayscale & prep inputs
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Could not read image at {img_path}")
    img_resized = cv2.resize(img_gray, target_size)
    img_input = preprocess_for_model(img_gray, target_size=target_size)

    # --- Grad-CAM ---
    hm_gc, preds_gc = generate_gradcam(model, img_input, layer_name=layer_name)
    ov_gc = overlay_heatmap(img_resized, hm_gc)

    # --- Score-CAM ---
    hm_sc, preds_sc, _ = generate_scorecam(
        model, img_input, layer_name=layer_name,
        channel_limit=scorecam_channel_limit
    )
    ov_sc = overlay_heatmap(img_resized, hm_sc)

    # Choose a single prediction summary (from Grad-CAM preds)
    pred_idx = int(np.argmax(preds_gc))
    pred_label = class_names[pred_idx]
    pred_conf = float(preds_gc[pred_idx] * 100.0)

    # Build true label text
    true_text = f"True: {class_names[true_label_idx]}" if true_label_idx is not None else "True: Unknown"

    # --- LIME ---
    img_rgb_for_lime = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_rgb_for_lime,
        make_lime_predict_fn(model),
        top_labels=1,
        hide_color=0,
        num_samples=lime_num_samples
    )
    top_class = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_class,
        positive_only=True,
        num_features=lime_num_features,
        hide_rest=False
    )
    lime_vis = mark_boundaries(temp, mask)

    # --- Plot 1x4 row ---
    plt.figure(figsize=(20, 5))  # wide row

    # 1) Original
    plt.subplot(1,4,1)
    plt.imshow(img_resized, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # 2) Grad-CAM
    plt.subplot(1,4,2)
    plt.imshow(ov_gc[..., ::-1])  # BGR->RGB
    plt.title(f"Grad-CAM\n{true_text}\nPred: {pred_label} ({pred_conf:.2f}%)")
    plt.axis("off")

    # 3) LIME
    plt.subplot(1,4,3)
    plt.imshow(lime_vis)
    plt.title(f"LIME (top: {class_names[top_class]})")
    plt.axis("off")

    # 4) Score-CAM
    plt.subplot(1,4,4)
    plt.imshow(ov_sc[..., ::-1])  # BGR->RGB
    plt.title("Score-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    print(f"Saved 4-panel row visualization to: {save_path}")
    return save_path, pred_label, pred_conf, pred_idx



# e.g. class_names from your dataset
class_names = train_ds_raw.class_names

save_path, pred_label, pred_conf, pred_idx = show_all_explanations(
    model=model,
    img_path="/kaggle/working/splited/test/00 Normal/00 (1001).jpg",  # <-- your image
    class_names=class_names,
    true_label=0,                          # index or string e.g. "Normal"
    layer_name=None,                       # auto-picks last Conv2D if None
    save_path="all_explanations.png",
    lime_num_samples=100,                  # tweak for stability/quality vs speed
    lime_num_features=10,
    scorecam_channel_limit=None,           # or e.g. 64 to speed up
    show=True
)