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
            # If layer name not found, find the last Conv2D layer
            last_conv_layer = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
    else:
        # Find the last Conv2D layer
        last_conv_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
    
    if last_conv_layer is None:
        print("No Conv2D layer found. Returning empty heatmap.")
        # Return an empty heatmap of appropriate size
        if len(img_array.shape) == 4:
            return np.zeros((img_array.shape[1], img_array.shape[2]))
        else:
            return np.zeros((1, 1))
    
    # Create a model that outputs both the last conv layer output and the final output
    if isinstance(model, tf.keras.Sequential):
        # For Sequential models, we need to recreate the model flow
        # First, create a model that outputs the conv layer activation
        conv_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=last_conv_layer.output
        )
        
        # Function to get gradients
        @tf.function
        def get_gradients_and_activations():
            # Forward pass through the model
            with tf.GradientTape() as tape:
                # Get the conv layer output
                conv_output = conv_model(img_array)
                tape.watch(conv_output)
                
                # Pass this through the rest of the model
                # Find the index of the conv layer
                conv_index = model.layers.index(last_conv_layer)
                
                # Apply remaining layers
                x = conv_output
                for i in range(conv_index + 1, len(model.layers)):
                    x = model.layers[i](x)
                
                # Get the target class score
                if isinstance(x, list):
                    target_output = x[0][:, pred_index]
                else:
                    target_output = x[:, pred_index]
                
                # Compute gradients
                grads = tape.gradient(target_output, conv_output)
                return grads, conv_output
        
    else:
        # For Functional models, we can use a more direct approach
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Function to get gradients
        @tf.function
        def get_gradients_and_activations():
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_array)
                tape.watch(conv_output)
                
                # Handle different output formats
                if isinstance(predictions, list):
                    target_output = predictions[0][:, pred_index]
                else:
                    target_output = predictions[:, pred_index]
                
                # Compute gradients
                grads = tape.gradient(target_output, conv_output)
                return grads, conv_output
    
    # Get gradients and activations
    try:
        grads, conv_output = get_gradients_and_activations()
        
        # Safe check for None gradients
        if grads is None:
            print("Gradient is None. Returning empty heatmap.")
            return np.zeros((img_array.shape[1], img_array.shape[2]))
            
        # Average gradients over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv layer outputs with the gradients
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Error computing Grad-CAM: {e}")
        # Return placeholder heatmap
        return np.zeros((img_array.shape[1], img_array.shape[2]))


def overlay_heatmap(orig_img: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a heatmap onto the original image.

    Args:
        orig_img: Original image array (H, W, 3), dtype=uint8.
        heatmap: 2D numpy array (h, w) with values in [0,1].
        alpha: Heatmap transparency factor.

    Returns:
        superimposed_img: Combined image as numpy uint8 array.
    """
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply the JET colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose heatmap on original image
    superimposed = cv2.addWeighted(heatmap_color, alpha, orig_img, 1 - alpha, 0)
    return superimposed


if __name__ == "__main__":
    import sys
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    if len(sys.argv) != 5:
        print("Usage: python grad_cam_utils.py <image_path> <model_path> "
              "<last_conv_layer_name or 'auto'> <output_path>")
        sys.exit(1)

    img_path, model_path, layer_name, out_path = sys.argv[1:]
    model = tf.keras.models.load_model(model_path)

    # Default preprocess: RGB 224x224 normalized
    img = load_img(img_path, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    last_conv = None if layer_name.lower() == 'auto' else layer_name
    heatmap = make_gradcam_heatmap(img_arr, model, last_conv)
    orig = cv2.imread(img_path)
    cam = overlay_heatmap(orig, heatmap)
    cv2.imwrite(out_path, cam)
    print(f"Saved Grad-CAM overlay to {out_path}")