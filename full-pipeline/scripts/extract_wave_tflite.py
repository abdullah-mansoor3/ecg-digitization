import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class WaveExtractor:
    def __init__(self, weights_path, device='cpu'):
        """
        Initialize the UNet model and set device.
        """
        print('initializing wave extractor...')
        self.model = tf.lite.Interpreter(model_path=weights_path)
        self.model.allocate_tensors()
        print('Wave extractor initialized.')

    def preprocess_for_model(self, img):
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        # Input size expected by model
        input_width = input_details[0]['shape'][3]
        input_height = input_details[0]['shape'][2]
        img_resized = img.resize((input_width, input_height))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        return img_array[np.newaxis, np.newaxis, :, :]  # shape: [1, 1, H, W]

    def extract_wave(self, cropped_lead_img_path):
        """
        Takes the path to a cropped lead image, returns the extracted wave mask as a numpy array.
        The output is center-cropped to remove any padding.
        """
        # Load raw original image
        original_img = Image.open(cropped_lead_img_path).convert("L")
        original_size = original_img.size  # (W, H)

        # Preprocess for model input
        input_tensor = self.preprocess_for_model(img=original_img)

        # Run inference
        self.model.set_tensor(self.model.get_input_details()[0]['index'], input_tensor)
        self.model.invoke()
        output_data = self.model.get_tensor(self.model.get_output_details()[0]['index'])
        pred_mask = output_data[0, 0, :, :]  # shape: (H, W)

        # Convert predicted mask to binary and resize to original size
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
        pred_mask_img = Image.fromarray(pred_mask_bin).resize(original_size, Image.BILINEAR)

        return pred_mask_img

        
    def plot_wave(self, binary_mask, title="Extracted Waveform Binary Mask"):
        """
        Plots the binary waveform mask.
        Args:
            binary_mask (np.ndarray): 2D binary mask (output of extract_wave)
            title (str): Plot title
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(binary_mask, cmap='gray', aspect='auto')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
