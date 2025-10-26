# import os
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# def land_water_detection(image_path, output_mask_path=None, show=False):
#     """
#     Detects land and water areas in a color image using K-means clustering.
#     Args:
#         image_path (str): Path to the input color image.
#         output_mask_path (str): Path to save the binary mask (optional).
#         show (bool): If True, display the mask and original image.
#     Returns:
#         np.ndarray: Binary mask (1=water, 0=land)
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"❌ Image not found: {image_path}")

#     # Convert BGR → RGB for processing
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     Z = img_rgb.reshape((-1, 3))
#     Z = np.float32(Z)

#     # K-means clustering (2 clusters: land and water)
#     K = 2
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     labels = labels.flatten()
#     centers = np.uint8(centers)
#     segmented_img = centers[labels].reshape(img_rgb.shape)

#     # Assume water is the darker cluster
#     cluster_means = np.mean(centers, axis=1)
#     water_cluster = np.argmin(cluster_means)
#     mask = (labels == water_cluster).astype(np.uint8).reshape(img_rgb.shape[:2])

#     # Optionally save the mask
#     if output_mask_path:
#         cv2.imwrite(output_mask_path, mask * 255)

#     # Optionally display results
#     if show:
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 3, 1)
#         plt.title('Original Image')
#         plt.imshow(img_rgb)
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.title('Segmented Image')
#         plt.imshow(segmented_img)
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.title('Water Mask')
#         plt.imshow(mask, cmap='gray')
#         plt.axis('off')
#         plt.tight_layout()
#         plt.show()

#     return mask


# def process_all_generated_images(input_dir, output_dir):
#     """Processes all images in a folder for land/water detection."""
#     os.makedirs(output_dir, exist_ok=True)
#     for fname in os.listdir(input_dir):
#         if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
#             in_path = os.path.join(input_dir, fname)
#             out_path = os.path.join(output_dir, f"mask_{fname}")
#             print(f"Processing {in_path} → {out_path}")
#             land_water_detection(in_path, out_path)


import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from land.utils import CLASS_RGB, colour_code_segmentation

def load_model(checkpoint_path, encoder='resnet50', device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=None,
        classes=len(CLASS_RGB),
        activation='sigmoid'
    )
    state = torch.load(checkpoint_path, map_location=device)
    # ensure all weights are float32
    for k in state:
        state[k] = state[k].float()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_mask_on_image(image_path, checkpoint, encoder='resnet50', preprocess_fn=None, device=None, target_size=(512,512)):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint, encoder=encoder, device=device)

    # read image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # resize to model input
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    inp = resized.astype('float32')

    # apply encoder preprocessing if available
    if preprocess_fn:
        inp = preprocess_fn(inp)

    # convert to tensor (1xCxHxW) and ensure float32
    inp_tensor = torch.from_numpy(inp.transpose(2,0,1)[None, ...]).float().to(device)

    with torch.no_grad():
        pred = model(inp_tensor)
        pred = pred.squeeze().cpu().numpy()  # HxWxC or CxHxW depending on activation

        # if model outputs multi-class (C channels)
        if pred.shape[0] == len(CLASS_RGB):
            pred = np.transpose(pred, (1,2,0))  # HxWxC

        class_map = np.argmax(pred, axis=-1)
        color_mask = colour_code_segmentation(class_map, CLASS_RGB)

    # resize mask back to original image size
    color_mask = cv2.resize(color_mask.astype('uint8'), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return color_mask, pred


if __name__ == '__main__':
    import argparse
    import segmentation_models_pytorch as smp
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--encoder', default='resnet50')
    args = parser.parse_args()

    # if you want the exact preprocessing fn for encoder:
    preprocess_fn = smp.encoders.get_preprocessing_fn(args.encoder, 'imagenet')
    mask, logits = predict_mask_on_image(args.image, args.checkpoint, encoder=args.encoder, preprocess_fn=preprocess_fn)
    out_path = os.path.splitext(args.image)[0] + '_pred.png'
    cv2.imwrite(out_path, mask[:,:,::-1])  # BGR write
    print("Saved:", out_path)
