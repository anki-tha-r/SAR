import cv2
import torch
import numpy as np
from unet import UNet

# Function to load the pre-trained model
def load_model(model_path='land_checkpoints/land_unet_model.pth'):
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Load model state dict
    model.eval()  # Set model to evaluation mode
    return model

# Function to predict the mask from an input image
def predict_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (256, 256))  # Resize image to 256x256 (input size for the model)
    
    # Convert to tensor, normalize, and add batch dimension
    image_input = torch.tensor(image_resized / 255.0, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW format

    # Load the model
    model = load_model()

    # Make prediction
    with torch.no_grad():
        output = model(image_input)  # Forward pass
        output = output[0][0].numpy()  # Get the output from the batch and channel dimension

    # Convert the output to a binary mask
    mask = (output > 0.5).astype(np.uint8) * 255  # Apply threshold for binary mask (0 or 255)

    # Resize the mask to match original image dimensions
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize mask to original image size
    return mask_resized

# Example usage (optional if running script directly)
if __name__ == '__main__':
    image_path = 'img-1.png'  # Replace with your image path
    output_mask = predict_image(image_path)
    cv2.imwrite('output_mask.png', output_mask)  # Save the output mask as a PNG
    print("Prediction complete and mask saved.")
