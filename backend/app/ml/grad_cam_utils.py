import sys
import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Compute Grad-CAM heatmap for a given image and model.
    Supports binary or multi-class outputs, works for Sequential and Functional.
    """
    # Ensure the model is built
    _ = model(img_array)

    # Locate last conv layer
    last_conv = None
    if last_conv_layer_name:
        try:
            last_conv = model.get_layer(last_conv_layer_name)
        except (ValueError, AttributeError):
            last_conv = None
    if last_conv is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break
    if last_conv is None:
        print("No Conv2D layer found. Returning zero heatmap.")
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    # For Sequential models, build separate conv_model
    if isinstance(model, tf.keras.Sequential):
        conv_model = tf.keras.Model(inputs=model.inputs, outputs=last_conv.output)
        layer_index = model.layers.index(last_conv)
        # GradientTape on conv output
        with tf.GradientTape() as tape:
            conv_output = conv_model(img_array)
            tape.watch(conv_output)
            # forward pass through remaining layers
            x = conv_output
            for layer in model.layers[layer_index+1:]:
                x = layer(x)
            preds = x
            # select target
            if preds.shape[-1] == 1:
                target = preds[:, 0]
            else:
                class_idx = tf.argmax(preds[0])
                target = preds[:, class_idx]
        grads = tape.gradient(target, conv_output)
    else:
        # Functional or subclassed model
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]   # use model.output here
        )
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)  # this will now give you [conv_outputs, symbolic preds]
            tape.watch(conv_output)
            if preds.shape[-1] == 1:
                target = preds[:, 0]
            else:
                class_idx = tf.argmax(preds[0])
                target = preds[:, class_idx]
        grads = tape.gradient(target, conv_output)

    if grads is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_output[0]

    # Weight activations and compute heatmap
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())
    heatmap /= max_val
    return heatmap.numpy()


def create_contoured_spot_heatmap(orig_img: np.ndarray,
                                  heatmap: np.ndarray,
                                  alpha: float = 0.5,
                                  threshold: float = 0.4,
                                  max_spots: int = 8,
                                  color_scheme: str = "hot",
                                  add_labels: bool = True,
                                  contour_thickness: int = 2,
                                  min_spot_area: int = 50,
                                  blur_size: int = 9,
                                  morph_kernel_size: int = 5,
                                  adaptive_threshold: bool = True) -> np.ndarray:
    """
    Visualize heatmap with contour spots.
    """
    result = orig_img.copy()
    h, w = orig_img.shape[:2]
    heat_resized = cv2.resize(heatmap, (w, h))
    smoothed = cv2.GaussianBlur(heat_resized, (blur_size, blur_size), 0)

    if adaptive_threshold:
        non_zero = smoothed[smoothed > 0]
        if non_zero.size:
            thr_adapt = np.percentile(non_zero, 70)
            threshold = float(np.clip(thr_adapt, 0.2, 0.7))

    mask = (smoothed > threshold).astype(np.uint8)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_spot_area]
    # sort by area*intensity
    def score(c):
        m = np.zeros_like(smoothed)
        cv2.drawContours(m, [c], -1, 1, -1)
        return cv2.contourArea(c) * np.max(smoothed * m)
    contours.sort(key=score, reverse=True)
    contours = contours[:max_spots]

    overlay = np.zeros_like(result)
    colormaps = {
        "hot": cv2.COLORMAP_HOT,
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS
    }
    cmap = colormaps.get(color_scheme, cv2.COLORMAP_HOT)

    for i, cnt in enumerate(contours):
        ellipse = cv2.fitEllipse(cnt)
        mask = np.zeros_like(smoothed)
        cv2.drawContours(mask, [cnt], -1, 1, -1)
        intensity = np.max(smoothed * mask)
        color_val = int(255 * intensity)
        color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cmap)[0,0].tolist()
        # draw filled
        cv2.ellipse(overlay, ellipse, color, -1)
        # draw outline
        cv2.ellipse(result, ellipse, color, contour_thickness)
        if add_labels:
            (x, y), _, _ = ellipse
            x, y = int(x), int(y)
            cv2.circle(result, (x, y), 12, (255,255,255), -1)
            cv2.putText(result, str(i+1), (x-4, y+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    cv2.addWeighted(overlay, alpha, result, 1, 0, result)
    return result


def overlay_heatmap(orig_img: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Simple overlay of heatmap on image.
    """
    h, w = orig_img.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_uint = np.uint8(255 * hm)
    colored = cv2.applyColorMap(hm_uint, cv2.COLORMAP_JET)
    return cv2.addWeighted(colored, alpha, orig_img, 1-alpha, 0)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python grad_cam_utils.py <img> <model> <layer> <out> [overlay]")
        sys.exit(1)
    img_path, model_path, layer_name, out_path = sys.argv[1:5]
    overlay_type = sys.argv[5] if len(sys.argv)>5 else "spots"
    model = tf.keras.models.load_model(model_path)
    img = cv2.imread(img_path)
    arr = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224,224)) / 255.0
    arr = np.expand_dims(arr,0)
    layer = None if layer_name.lower()=="auto" else layer_name
    hm = make_gradcam_heatmap(arr, model, layer)
    res = overlay_heatmap(img, hm) if overlay_type=="overlay" else create_contoured_spot_heatmap(img, hm)
    cv2.imwrite(out_path, res)
    print(f"Saved to {out_path}")
