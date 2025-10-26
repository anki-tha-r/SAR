import os
import re
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import importlib
import torch.nn.functional as F

# try to import rasterio for robust tiff handling, but keep optional
try:
    import rasterio
except Exception:
    rasterio = None

# try segmentation_models_pytorch
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


def _load_label_colors():
    global _label_colors, _num_classes
    if _label_colors is not None:
        return _label_colors
    if os.path.exists(LABEL_CSV):
        df = pd.read_csv(LABEL_CSV)
        if all(c in df.columns for c in ("r", "g", "b")):
            colors = [tuple(map(int, row)) for row in df[["r", "g", "b"]].values]
        else:
            # fallback if single column
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
        # common keys
        for k in ("model_state_dict", "state_dict"):
            if k in ckpt:
                return ckpt[k]
        return ckpt  # may already be state_dict
    return ckpt


def _infer_in_channels_from_state(state_dict):
    # look for first conv weight with 4 dims and sensible in_channels
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
    if smp is None:
        raise RuntimeError("segmentation_models_pytorch not available (install via pip)")

    ckpt_path = _find_latest_checkpoint()
    state = _load_checkpoint_state(ckpt_path)
    in_channels = _infer_in_channels_from_state(state)
    colors = _load_label_colors()
    classes = len(colors)

    # try common encoder names; prefer resnet34
    encoders = ["resnet34", "resnet18", "efficientnet-b0", "resnet50", "mobilenet_v2"]
    last_exc = None
    for enc in encoders:
        try:
            model = smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=in_channels, classes=classes)
            # clean keys
            new_state = {}
            for k, v in state.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                new_state[new_k] = v
            model.load_state_dict(new_state, strict=False)
            model.to(_device)
            model.eval()
            _model = model
            return _model
        except Exception as e:
            last_exc = e
            continue

    # final fallback: try resnet34 with strict=False
    try:
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels, classes=classes)
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
        model.to(_device)
        model.eval()
        _model = model
        return _model
    except Exception as e:
        raise RuntimeError(f"Failed to build/load model from {ckpt_path}") from (last_exc or e)


def _read_image(path):
    # Returns image as H,W,C (uint8/uint16/float32)
    ext = os.path.splitext(path)[1].lower()
    if rasterio and ext in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
            arr = np.transpose(arr, (1, 2, 0))
            return arr
    # fallback to cv2 for png/jpg or simple tiff
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image {path}")
    # cv2 returns H,W (gray) or H,W,C (BGR)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # convert BGR->RGB for consistency with label colors mapping later
    if img.shape[2] >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _to_tensor(img, expected_ch):
    # img: H,W,C (RGB) or H,W (gray)
    if img.ndim == 2:
        img = np.stack([img], axis=-1)
    h, w, ch = img.shape
    # select or pad channels
    if ch < expected_ch:
        # repeat last channel
        reps = expected_ch - ch
        pad = np.repeat(img[..., -1:], reps, axis=-1)
        img = np.concatenate([img, pad], axis=-1)
    elif ch > expected_ch:
        img = img[..., :expected_ch]

    # dtype scaling
    if np.issubdtype(img.dtype, np.uint8):
        img_f = img.astype(np.float32) / 255.0
    elif np.issubdtype(img.dtype, np.uint16):
        img_f = img.astype(np.float32) / 65535.0
    elif np.issubdtype(img.dtype, np.floating):
        img_f = img.astype(np.float32)
        # clamp to [0,1]
        img_f = np.clip(img_f, 0.0, 1.0)
    else:
        img_f = img.astype(np.float32)
        img_f = img_f / img_f.max() if img_f.max() > 0 else img_f

    # HWC -> CHW
    tensor = torch.from_numpy(np.transpose(img_f, (2, 0, 1))).unsqueeze(0).to(_device)
    return tensor


def _postprocess_preds(preds, colors):
    # preds: H,W int class indices
    h, w = preds.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, col in enumerate(colors):
        color_mask[preds == idx] = col  # RGB
    # convert to BGR for cv2
    return cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)


def _run_inference_on_tensor(model, tensor, out_size):
    with torch.no_grad():
        out = model(tensor)  # (1,classes,Hout,Wout)
        out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
        probs = torch.softmax(out, dim=1)
        preds = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return preds


def predict_image(input_path):
    """
    Main entry called by app.py
    Returns: BGR uint8 mask image (H,W,3) ready for cv2.imwrite
    """
    colors = _load_label_colors()
    model = _build_and_load_model()

    img = _read_image(input_path)
    orig_h, orig_w = img.shape[:2]

    # infer model expected in_channels from model's first conv if possible
    expected_ch = None
    try:
        # try to inspect model's encoder first conv weight
        for name, param in model.named_parameters():
            if name.endswith(".weight") and param.ndim == 4:
                expected_ch = param.shape[1]
                break
    except Exception:
        expected_ch = None
    if expected_ch is None:
        expected_ch = 3

    # try primary preprocess
    tensor = _to_tensor(img, expected_ch)
    preds = _run_inference_on_tensor(model, tensor, out_size=(orig_h, orig_w))

    # if prediction is nearly empty (very few building pixels) try alternate scaling
    positive_ratio = (preds > 0).sum() / (orig_h * orig_w)
    if positive_ratio < 0.001:
        # retry with alternative scaling for uint16 images (some tiff were 16-bit)
        if np.issubdtype(img.dtype, np.uint16):
            # convert by scaling with 255 (if training used 8-bit conversion)
            img_alt = (img.astype(np.float32) / 255.0).astype(np.float32)
            tensor_alt = _to_tensor(img_alt, expected_ch)
            preds_alt = _run_inference_on_tensor(model, tensor_alt, out_size=(orig_h, orig_w))
            positive_ratio_alt = (preds_alt > 0).sum() / (orig_h * orig_w)
            if positive_ratio_alt > positive_ratio:
                preds = preds_alt
                positive_ratio = positive_ratio_alt

        # final fallback: attempt histogram equalization on luminance
        try:
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            img_eq = cv2.equalizeHist(img_gray)
            img_eq_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            tensor_eq = _to_tensor(img_eq_rgb, expected_ch)
            preds_eq = _run_inference_on_tensor(model, tensor_eq, out_size=(orig_h, orig_w))
            positive_ratio_eq = (preds_eq > 0).sum() / (orig_h * orig_w)
            if positive_ratio_eq > positive_ratio:
                preds = preds_eq
        except Exception:
            pass

    mask_bgr = _postprocess_preds(preds, colors)
    return mask_bgr
# ...existing code...