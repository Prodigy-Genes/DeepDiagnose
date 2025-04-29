import sys
import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    A simplified and robust Grad-CAM implementation that works with different model types.
    
    Args:
        img_array: Preprocessed image array of shape (1, H, W, C).
        model: A Keras model instance.
        last_conv_layer_name: Optional name of the final conv layer.
        
    Returns:
        heatmap: 2D numpy array with values in [0,1].
    """
    # Run the model once to ensure it's built
    model_output = model(img_array)
    
    # Find the target class (highest probability)
    if isinstance(model_output, list):
        pred_index = tf.argmax(model_output[0][0])
    else:
        pred_index = tf.argmax(model_output[0])
    
    # Find the last conv layer
    if last_conv_layer_name:
        try:
            last_conv_layer = model.get_layer(last_conv_layer_name)
        except:
            last_conv_layer = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
    else:
        last_conv_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
    
    if last_conv_layer is None:
        print("No Conv2D layer found. Returning empty heatmap.")
        if len(img_array.shape) == 4:
            return np.zeros((img_array.shape[1], img_array.shape[2]))
        else:
            return np.zeros((1, 1))
    
    # Build gradient-taped model
    if isinstance(model, tf.keras.Sequential):
        conv_model = tf.keras.Model(inputs=model.inputs, outputs=last_conv_layer.output)
        @tf.function
        def get_gradients_and_activations():
            with tf.GradientTape() as tape:
                conv_output = conv_model(img_array)
                tape.watch(conv_output)
                conv_index = model.layers.index(last_conv_layer)
                x = conv_output
                for i in range(conv_index + 1, len(model.layers)):
                    x = model.layers[i](x)
                if isinstance(x, list):
                    target_output = x[0][:, pred_index]
                else:
                    target_output = x[:, pred_index]
                grads = tape.gradient(target_output, conv_output)
                return grads, conv_output
    else:
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
        @tf.function
        def get_gradients_and_activations():
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_array)
                tape.watch(conv_output)
                if isinstance(predictions, list):
                    target_output = predictions[0][:, pred_index]
                else:
                    target_output = predictions[:, pred_index]
                grads = tape.gradient(target_output, conv_output)
                return grads, conv_output
    
    try:
        grads, conv_output = get_gradients_and_activations()
        if grads is None:
            print("Gradient is None. Returning empty heatmap.")
            return np.zeros((img_array.shape[1], img_array.shape[2]))
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        return heatmap.numpy()
    except Exception as e:
        print(f"Error computing Grad-CAM: {e}")
        return np.zeros((img_array.shape[1], img_array.shape[2]))


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
    Creates a more advanced visualization with contoured ovals/circles and optional labels.
    
    Args:
        orig_img: Original image as a numpy array (H, W, C).
        heatmap: Generated Grad-CAM heatmap (2D array with values in [0,1]).
        alpha: Transparency level for the overlay (0-1).
        threshold: Minimum value to consider in heatmap (0-1).
        max_spots: Maximum number of spots to display.
        color_scheme: Color palette to use ("hot", "cool", "rainbow", "viridis").
        add_labels: Whether to add numbered labels to the spots.
        contour_thickness: Thickness of contour lines.
        min_spot_area: Minimum area (in pixels) for a spot to be considered valid.
        blur_size: Size of Gaussian blur kernel for smoothing.
        morph_kernel_size: Size of morphological operation kernel.
        adaptive_threshold: Whether to use adaptive thresholding based on heatmap distribution.
        
    Returns:
        Annotated image with contoured spots.
    """
    result = orig_img.copy()
    h, w = orig_img.shape[:2]
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Apply Gaussian blur for smoother contours
    smoothed = cv2.GaussianBlur(heatmap_resized, (blur_size, blur_size), 0)
    
    # Use adaptive thresholding if enabled
    if adaptive_threshold:
        # Calculate threshold based on heatmap distribution
        sorted_values = np.sort(smoothed.flatten())
        # Use the value at 70th percentile of non-zero values as threshold
        non_zero_values = sorted_values[sorted_values > 0]
        if len(non_zero_values) > 0:
            adaptive_thresh = np.percentile(non_zero_values, 70)
            # Ensure threshold is at least 0.2 and not more than 0.7
            threshold = max(0.2, min(adaptive_thresh, 0.7))
    
    # Create binary mask
    binary = (smoothed > threshold).astype(np.uint8)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_spot_area]
    
    # Sort contours by importance (area * max intensity)
    def contour_importance(contour):
        mask = np.zeros_like(smoothed)
        cv2.drawContours(mask, [contour], 0, 1, -1)
        intensity = np.max(smoothed * mask)
        area = cv2.contourArea(contour)
        return area * intensity
    
    valid_contours.sort(key=contour_importance, reverse=True)
    valid_contours = valid_contours[:max_spots]
    
    # Create overlay for the spots
    overlay = np.zeros_like(result)
    
    # Define color maps
    color_maps = {
        "hot": [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 128)],
        "cool": [(255, 0, 0), (255, 0, 128), (255, 0, 255), (128, 0, 255)],
        "rainbow": [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)],
        "viridis": [(68, 1, 84), (59, 82, 139), (33, 144, 140), (93, 201, 99), (253, 231, 37)]
    }
    
    colors = color_maps.get(color_scheme, color_maps["hot"])
    
    # Draw contours and labels
    for i, contour in enumerate(valid_contours):
        if len(contour) >= 5:  # Minimum points needed for ellipse fitting
            # Fit ellipse to contour
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            x, y = int(center[0]), int(center[1])
            
            # Compute spot intensity
            mask = np.zeros_like(smoothed)
            cv2.drawContours(mask, [contour], 0, 1, -1)
            intensity = np.max(smoothed * mask)
            
            # Determine color based on intensity
            color_idx = min(int(intensity * len(colors)), len(colors) - 1)
            color = colors[color_idx]
            
            # Draw filled ellipse on overlay
            cv2.ellipse(overlay, ellipse, color, -1)
            
            # Draw ellipse contour on result
            cv2.ellipse(result, ellipse, color, contour_thickness)
            
            # Add label if requested
            if add_labels:
                # Create white circle background for number
                cv2.circle(result, (x, y), 12, (255, 255, 255), -1)
                # Add spot number
                cv2.putText(result, str(i+1), (x-4, y+4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Blend overlay with result
    cv2.addWeighted(overlay, alpha, result, 1, 0, result)
    
    # Add legend if labels are enabled and spots were found
    if add_labels and valid_contours:
        legend_height = 40
        canvas = np.ones((result.shape[0] + legend_height, result.shape[1], 3), dtype=np.uint8) * 240
        canvas[:result.shape[0], :result.shape[1]] = result
        cv2.putText(canvas, "Spots indicate areas of model focus, ranked by importance",
                   (10, result.shape[0] + 25), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas
    
    return result


def overlay_heatmap(orig_img: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Simple overlay of heatmap on original image.
    """
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (5, 5), 0)
    heatmap_resized = np.where(heatmap_resized > 0.3, heatmap_resized, 0)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap_color, alpha, orig_img, 1 - alpha, 0)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python grad_cam_utils.py <image_path> <model_path> "
              "<last_conv_layer_name or 'auto'> <output_path> [visualization_type]")
        print("visualization_type options: 'spots' (default) or 'overlay'")
        sys.exit(1)

    img_path, model_path, layer_name, out_path = sys.argv[1:5]
    viz_type = sys.argv[5] if len(sys.argv) > 5 else "spots"
    
    model = tf.keras.models.load_model(model_path)

    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(img_path, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    last_conv = None if layer_name.lower() == 'auto' else layer_name
    heatmap = make_gradcam_heatmap(img_arr, model, last_conv)
    orig = cv2.imread(img_path)
    
    if viz_type.lower() == "overlay":
        result = overlay_heatmap(orig, heatmap)
    else:  # Default to spots
        result = create_contoured_spot_heatmap(
            orig, 
            heatmap,
            alpha=0.5,
            threshold=0.4,
            adaptive_threshold=True
        )
    
    cv2.imwrite(out_path, result)
    print(f"Saved Grad-CAM visualization to {out_path}")