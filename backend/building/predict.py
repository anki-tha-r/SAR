# import os
# import re
# import glob
# import cv2
# import torch
# import numpy as np
# import pandas as pd
# import importlib
# import torch.nn.functional as F

# # try to import rasterio for robust tiff handling, but keep optional
# try:
#     import rasterio
# except Exception:
#     rasterio = None

# # try segmentation_models_pytorch
# try:
#     import segmentation_models_pytorch as smp
# except Exception:
#     smp = None

# BUILDING_DIR = os.path.dirname(__file__)
# CHECKPOINT_DIR = os.path.join(BUILDING_DIR, "checkpoints")
# LABEL_CSV = os.path.join(BUILDING_DIR, "label_class_dict.csv")

# _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# _model = None
# _label_colors = None
# _num_classes = None

# # ...existing code...

# def one_hot_encode(label, label_values):
#     """
#     Convert an RGB label (H,W,3) to one-hot (H,W,num_classes) using label_values (list of [r,g,b]).
#     """
#     semantic_map = []
#     for colour in label_values:
#         equality = np.equal(label, colour)
#         class_map = np.all(equality, axis=-1)
#         semantic_map.append(class_map)
#     semantic_map = np.stack(semantic_map, axis=-1).astype(np.uint8)
#     return semantic_map

# def reverse_one_hot(one_hot_image):
#     """
#     Convert one-hot (H,W,num_classes) -> single channel class indices (H,W).
#     """
#     return np.argmax(one_hot_image, axis=-1).astype(np.uint8)

# def colour_code_segmentation(class_idx_image, label_values):
#     """
#     Map single-channel class index image -> RGB image using label_values (list/array of [r,g,b]).
#     """
#     colour_codes = np.array(label_values, dtype=np.uint8)
#     # class_idx_image contains integers 0..num_classes-1
#     rgb = colour_codes[class_idx_image.astype(int)]
#     return rgb  # RGB (H,W,3) uint8

# # Optionally replace or keep existing _postprocess_preds; here's a version using colour_code_segmentation
# def _postprocess_preds(preds, colors):
#     """
#     preds: H,W class indices
#     colors: list of (r,g,b)
#     returns: BGR uint8 mask suitable for cv2.imwrite
#     """
#     rgb_mask = colour_code_segmentation(preds, colors)  # RGB
#     bgr_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
#     return bgr_mask

# # ...existing code...

# def _load_label_colors():
#     global _label_colors, _num_classes
#     if _label_colors is not None:
#         return _label_colors
#     if os.path.exists(LABEL_CSV):
#         df = pd.read_csv(LABEL_CSV)
#         if all(c in df.columns for c in ("r", "g", "b")):
#             colors = [tuple(map(int, row)) for row in df[["r", "g", "b"]].values]
#         else:
#             # fallback if single column
#             colors = [(0, 0, 0), (255, 255, 255)]
#     else:
#         colors = [(0, 0, 0), (255, 255, 255)]
#     _label_colors = colors
#     _num_classes = len(colors)
#     return _label_colors


# def _find_latest_checkpoint():
#     files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
#     if not files:
#         raise FileNotFoundError(f"No .pth files found in {CHECKPOINT_DIR}")
#     def epoch_key(p):
#         m = re.search(r"(\d+)(?=\.pth$)", os.path.basename(p))
#         if m:
#             return int(m.group(1))
#         return int(os.path.getmtime(p))
#     files.sort(key=epoch_key, reverse=True)
#     return files[0]


# def _load_checkpoint_state(ckpt_path):
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     if isinstance(ckpt, dict):
#         # common keys
#         for k in ("model_state_dict", "state_dict"):
#             if k in ckpt:
#                 return ckpt[k]
#         return ckpt  # may already be state_dict
#     return ckpt


# def _infer_in_channels_from_state(state_dict):
#     # look for first conv weight with 4 dims and sensible in_channels
#     for k, v in state_dict.items():
#         if hasattr(v, "ndim") and v.ndim == 4:
#             in_ch = v.shape[1]
#             if in_ch in (1, 2, 3, 4, 8):
#                 return int(in_ch)
#     return 3


# def _build_and_load_model():
#     global _model
#     if _model is not None:
#         return _model
#     if smp is None:
#         raise RuntimeError("segmentation_models_pytorch not available (install via pip)")

#     ckpt_path = _find_latest_checkpoint()
#     state = _load_checkpoint_state(ckpt_path)
#     in_channels = _infer_in_channels_from_state(state)
#     colors = _load_label_colors()
#     classes = len(colors)

#     # try common encoder names; prefer resnet34
#     encoders = ["resnet34", "resnet18", "efficientnet-b0", "resnet50", "mobilenet_v2"]
#     last_exc = None
#     for enc in encoders:
#         try:
#             model = smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=in_channels, classes=classes)
#             # clean keys
#             new_state = {}
#             for k, v in state.items():
#                 new_k = k.replace("module.", "") if k.startswith("module.") else k
#                 new_state[new_k] = v
#             model.load_state_dict(new_state, strict=False)
#             model.to(_device)
#             model.eval()
#             _model = model
#             return _model
#         except Exception as e:
#             last_exc = e
#             continue

#     # final fallback: try resnet34 with strict=False
#     try:
#         model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels, classes=classes)
#         new_state = {k.replace("module.", ""): v for k, v in state.items()}
#         model.load_state_dict(new_state, strict=False)
#         model.to(_device)
#         model.eval()
#         _model = model
#         return _model
#     except Exception as e:
#         raise RuntimeError(f"Failed to build/load model from {ckpt_path}") from (last_exc or e)


# def _read_image(path):
#     # Returns image as H,W,C (uint8/uint16/float32)
#     ext = os.path.splitext(path)[1].lower()
#     if rasterio and ext in (".tif", ".tiff"):
#         with rasterio.open(path) as src:
#             arr = src.read()  # (bands, H, W)
#             arr = np.transpose(arr, (1, 2, 0))
#             return arr
#     # fallback to cv2 for png/jpg or simple tiff
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     if img is None:
#         raise FileNotFoundError(f"Could not read image {path}")
#     # cv2 returns H,W (gray) or H,W,C (BGR)
#     if img.ndim == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     # convert BGR->RGB for consistency with label colors mapping later
#     if img.shape[2] >= 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img


# def _to_tensor(img, expected_ch):
#     # img: H,W,C (RGB) or H,W (gray)
#     if img.ndim == 2:
#         img = np.stack([img], axis=-1)
#     h, w, ch = img.shape
#     # select or pad channels
#     if ch < expected_ch:
#         # repeat last channel
#         reps = expected_ch - ch
#         pad = np.repeat(img[..., -1:], reps, axis=-1)
#         img = np.concatenate([img, pad], axis=-1)
#     elif ch > expected_ch:
#         img = img[..., :expected_ch]

#     # dtype scaling
#     if np.issubdtype(img.dtype, np.uint8):
#         img_f = img.astype(np.float32) / 255.0
#     elif np.issubdtype(img.dtype, np.uint16):
#         img_f = img.astype(np.float32) / 65535.0
#     elif np.issubdtype(img.dtype, np.floating):
#         img_f = img.astype(np.float32)
#         # clamp to [0,1]
#         img_f = np.clip(img_f, 0.0, 1.0)
#     else:
#         img_f = img.astype(np.float32)
#         img_f = img_f / img_f.max() if img_f.max() > 0 else img_f

#     # HWC -> CHW
#     tensor = torch.from_numpy(np.transpose(img_f, (2, 0, 1))).unsqueeze(0).to(_device)
#     return tensor


# # def _postprocess_preds(preds, colors):
# #     # preds: H,W int class indices
# #     h, w = preds.shape
# #     color_mask = np.zeros((h, w, 3), dtype=np.uint8)
# #     for idx, col in enumerate(colors):
# #         color_mask[preds == idx] = col  # RGB
# #     # convert to BGR for cv2
# #     return cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
# def _postprocess_preds(preds, colors):
#     """
#     preds: H,W class indices
#     colors: list of (r,g,b)
#     returns: BGR uint8 mask suitable for cv2.imwrite
#     """
#     rgb_mask = colour_code_segmentation(preds, colors)  # RGB
#     bgr_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
#     return bgr_mask


# def _run_inference_on_tensor(model, tensor, out_size):
#     with torch.no_grad():
#         out = model(tensor)  # (1,classes,Hout,Wout)
#         out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
#         probs = torch.softmax(out, dim=1)
#         preds = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
#         return preds


# def predict_image(input_path):
#     """
#     Main entry called by app.py
#     Returns: BGR uint8 mask image (H,W,3) ready for cv2.imwrite
#     """
#     colors = _load_label_colors()
#     model = _build_and_load_model()

#     img = _read_image(input_path)
#     orig_h, orig_w = img.shape[:2]

#     # infer model expected in_channels from model's first conv if possible
#     expected_ch = None
#     try:
#         # try to inspect model's encoder first conv weight
#         for name, param in model.named_parameters():
#             if name.endswith(".weight") and param.ndim == 4:
#                 expected_ch = param.shape[1]
#                 break
#     except Exception:
#         expected_ch = None
#     if expected_ch is None:
#         expected_ch = 3

#     # try primary preprocess
#     tensor = _to_tensor(img, expected_ch)
#     preds = _run_inference_on_tensor(model, tensor, out_size=(orig_h, orig_w))

#     # if prediction is nearly empty (very few building pixels) try alternate scaling
#     positive_ratio = (preds > 0).sum() / (orig_h * orig_w)
#     if positive_ratio < 0.001:
#         # retry with alternative scaling for uint16 images (some tiff were 16-bit)
#         if np.issubdtype(img.dtype, np.uint16):
#             # convert by scaling with 255 (if training used 8-bit conversion)
#             img_alt = (img.astype(np.float32) / 255.0).astype(np.float32)
#             tensor_alt = _to_tensor(img_alt, expected_ch)
#             preds_alt = _run_inference_on_tensor(model, tensor_alt, out_size=(orig_h, orig_w))
#             positive_ratio_alt = (preds_alt > 0).sum() / (orig_h * orig_w)
#             if positive_ratio_alt > positive_ratio:
#                 preds = preds_alt
#                 positive_ratio = positive_ratio_alt

#         # final fallback: attempt histogram equalization on luminance
#         try:
#             img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
#             img_eq = cv2.equalizeHist(img_gray)
#             img_eq_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
#             tensor_eq = _to_tensor(img_eq_rgb, expected_ch)
#             preds_eq = _run_inference_on_tensor(model, tensor_eq, out_size=(orig_h, orig_w))
#             positive_ratio_eq = (preds_eq > 0).sum() / (orig_h * orig_w)
#             if positive_ratio_eq > positive_ratio:
#                 preds = preds_eq
#         except Exception:
#             pass

#     mask_bgr = _postprocess_preds(preds, colors)
#     return mask_bgr
# # ...existing code...

import os
import re
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

# optional imports
try:
    import rasterio
except Exception:
    rasterio = None

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None

BUILDING_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BUILDING_DIR, "checkpoints")
LABEL_CSV = os.path.join(BUILDING_DIR, "label_class_dict.csv")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_label_colors = None
_num_classes = None

# -----------------------
# Helper functions (from your notebook)
# -----------------------
def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.uint8)
    return semantic_map

def reverse_one_hot(one_hot_image):
    return np.argmax(one_hot_image, axis=-1).astype(np.uint8)

def colour_code_segmentation(class_idx_image, label_values):
    colour_codes = np.array(label_values, dtype=np.uint8)
    rgb = colour_codes[class_idx_image.astype(int)]
    return rgb  # RGB (H,W,3) uint8

# -----------------------
# Model utilities
# -----------------------
def _load_label_colors():
    global _label_colors, _num_classes
    if _label_colors is not None:
        return _label_colors
    if os.path.exists(LABEL_CSV):
        df = pd.read_csv(LABEL_CSV)
        if all(c in df.columns for c in ("r", "g", "b")):
            colors = [tuple(map(int, row)) for row in df[["r", "g", "b"]].values]
        else:
            colors = [(0, 0, 0), (255, 255, 255)]
    else:
        colors = [(0, 0, 0), (255, 255, 255)]
    _label_colors = colors
    _num_classes = len(colors)
    return _label_colors

def _find_latest_checkpoint():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    if not files:
        raise FileNotFoundError(f"No .pth files found in {CHECKPOINT_DIR}")
    def epoch_key(p):
        m = re.search(r"(\d+)(?=\.pth$)", os.path.basename(p))
        if m:
            return int(m.group(1))
        return int(os.path.getmtime(p))
    files.sort(key=epoch_key, reverse=True)
    return files[0]

def _load_checkpoint_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict"):
            if k in ckpt:
                return ckpt[k]
        return ckpt
    return ckpt

def _clean_state_dict(state):
    new = {}
    for k, v in state.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new[new_k] = v
    return new

# Recreate the UNet from your notebook (so checkpoint compatibility is robust)
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNetCustom(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNetCustom, self).__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.maxpool(c1))
        c3 = self.down3(self.maxpool(c2))
        c4 = self.down4(self.maxpool(c3))
        bottleneck = self.bottleneck(self.maxpool(c4))
        u1 = self.up1(bottleneck)
        u1 = self.conv1(torch.cat([u1, c4], dim=1))
        u2 = self.up2(u1)
        u2 = self.conv2(torch.cat([u2, c3], dim=1))
        u3 = self.up3(u2)
        u3 = self.conv3(torch.cat([u3, c2], dim=1))
        u4 = self.up4(u3)
        u4 = self.conv4(torch.cat([u4, c1], dim=1))
        return torch.sigmoid(self.final_conv(u4)) if self.final_conv.out_channels == 1 else self.final_conv(u4)

def _infer_in_channels_from_state(state_dict):
    for k, v in state_dict.items():
        if hasattr(v, "ndim") and v.ndim == 4:
            in_ch = v.shape[1]
            if in_ch in (1, 2, 3, 4, 8):
                return int(in_ch)
    return 3

def _build_and_load_model():
    global _model
    if _model is not None:
        return _model
    ckpt_path = _find_latest_checkpoint()
    state = _load_checkpoint_state(ckpt_path)
    clean_state = _clean_state_dict(state)
    in_ch = _infer_in_channels_from_state(clean_state)
    colors = _load_label_colors()
    classes = len(colors)

    # Try custom UNet (notebook)
    try:
        # detect if custom UNet likely (single-channel outputs => n_classes==1)
        # If classes == 2 (background+building) notebook used n_classes=1 (binary)
        out_channels = 1 if classes == 2 else classes
        model = UNetCustom(n_channels=in_ch, n_classes=out_channels)
        model.load_state_dict(clean_state, strict=False)
        model.to(_device)
        model.eval()
        _model = model
        return _model
    except Exception:
        pass

    # Try segmentation_models_pytorch Unet variants
    if smp is None:
        raise RuntimeError("No suitable model class found and segmentation_models_pytorch unavailable.")
    encoders = ["resnet34", "resnet18", "efficientnet-b0", "resnet50"]
    for enc in encoders:
        try:
            model = smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=in_ch, classes=classes)
            model.load_state_dict(clean_state, strict=False)
            model.to(_device)
            model.eval()
            _model = model
            return _model
        except Exception:
            continue

    raise RuntimeError(f"Failed to build/load model from checkpoint {ckpt_path}")

# -----------------------
# Image read / preprocessing (pad/crop like notebook)
# -----------------------
def _read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if rasterio and ext in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
            arr = np.transpose(arr, (1, 2, 0))
            # convert to uint8-like range if necessary
            return arr
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _pad_to_multiple(img, multiple=32):
    h, w = img.shape[:2]
    size = max(h, w)
    # find smallest multiple >= size
    target = int(np.ceil(size / multiple) * multiple)
    pad_h = target - h
    pad_w = target - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
    return img_padded, (top, bottom, left, right)

def _remove_padding(img, pads):
    top, bottom, left, right = pads
    h, w = img.shape[:2]
    return img[top:h - bottom, left:w - right]

def _to_tensor(img, expected_ch):
    # img: H,W,C in RGB with values in [0,255] or float [0,1]
    if img.ndim == 2:
        img = np.stack([img], axis=-1)
    h, w, ch = img.shape
    if ch < expected_ch:
        reps = expected_ch - ch
        pad = np.repeat(img[..., -1:], reps, axis=-1)
        img = np.concatenate([img, pad], axis=-1)
    elif ch > expected_ch:
        img = img[..., :expected_ch]

    if np.issubdtype(img.dtype, np.uint8):
        img_f = img.astype(np.float32) / 255.0
    elif np.issubdtype(img.dtype, np.uint16):
        img_f = img.astype(np.float32) / 65535.0
    else:
        img_f = img.astype(np.float32)
        img_f = np.clip(img_f, 0.0, 1.0)

    tensor = torch.from_numpy(np.transpose(img_f, (2, 0, 1))).unsqueeze(0).to(_device)
    return tensor

def _postprocess_preds_to_bgr(preds, colors):
    # preds: H,W class indices (0..C-1) or binary 0/1
    rgb = colour_code_segmentation(preds, colors)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

# -----------------------
# Inference helpers
# -----------------------
def _run_model_and_get_preds(model, tensor, out_size, classes):
    with torch.no_grad():
        out = model(tensor)  # (1, out_channels, Hout, Wout)
        out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
        out_np = out.squeeze(0).cpu().numpy()  # (C, H, W) or (H, W) for single channel
        if out_np.ndim == 2:
            # single channel (binary sigmoid likely)
            probs = out_np
            preds = (probs > 0.5).astype(np.uint8)
            return preds
        # multi-channel
        if out_np.shape[0] == 1:
            probs = out_np[0]
            preds = (probs > 0.5).astype(np.uint8)
            return preds
        # shape: (C, H, W)
        probs_ch = np.transpose(out_np, (1, 2, 0))  # H W C
        # if multi-class, choose argmax
        preds = np.argmax(probs_ch, axis=-1).astype(np.uint8)
        return preds

# -----------------------
# Main entry used by app.py
# -----------------------
def predict_image(input_path):
    """
    Reads image at input_path, runs model and returns a BGR uint8 mask (H,W,3) ready for cv2.imwrite.
    Workflow:
      - read image (rasterio or cv2) -> RGB numpy
      - pad to square multiple of 32 (centered)
      - convert to tensor (0..1)
      - run model, get preds (class indices)
      - remove padding, colorize using label_class_dict.csv
    """
    colors = _load_label_colors()
    model = _build_and_load_model()

    img = _read_image(input_path)
    orig_h, orig_w = img.shape[:2]

    # pad image to multiple of 32 (ensures UNet pooling works)
    img_padded, pads = _pad_to_multiple(img, multiple=32)
    # infer expected channels from model first conv param if present
    expected_ch = 3
    try:
        for name, p in model.named_parameters():
            if name.endswith(".weight") and p.ndim == 4:
                expected_ch = int(p.shape[1])
                break
    except Exception:
        expected_ch = 3

    tensor = _to_tensor(img_padded, expected_ch)
    preds_padded = _run_model_and_get_preds(model, tensor, out_size=(img_padded.shape[0], img_padded.shape[1]), classes=len(colors))

    # crop/remove padding
    preds_cropped = _remove_padding(preds_padded, pads)

    # if predictions are binary (0/1), ensure dtype fits color mapping (0..C-1)
    if preds_cropped.ndim == 2 and preds_cropped.max() <= 1 and len(colors) > 2:
        # if model produced binary but colors > 2, map 1 -> building index 1 (assumption)
        pass

    mask_bgr = _postprocess_preds_to_bgr(preds_cropped, colors)
    return mask_bgr

# end of file