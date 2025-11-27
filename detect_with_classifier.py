#!/usr/bin/env python3
# Corrected sliding-window + pyramid detector for ViT models
# Compatible with MyViT input: (3, 32, 32)
#Code is experimental, might run into issues.

import argparse
import ast
import json
import time
from collections import OrderedDict

import cv2
import imutils
import numpy as np
import torch
from torchvision import transforms
from imutils.object_detection import non_max_suppression

from vit_model import MyViT   # your ViT model


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def sliding_window(image, step, ws):
    # ws = (w, h)
    for y in range(0, max(1, image.shape[0] - ws[1] + 1), step):
        for x in range(0, max(1, image.shape[1] - ws[0] + 1), step):
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])


def image_pyramid(image, scale=1.5, minSize=(32, 32)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


def safe_load_state(model, path, device):
    checkpoint = torch.load(path, map_location=device)

    # unwrap if "state_dict" exists
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # strip DP prefix "module."
    if any(k.startswith("module.") for k in checkpoint.keys()):
        new_state = OrderedDict()
        for k, v in checkpoint.items():
            new_state[k.replace("module.", "")] = v
        checkpoint = new_state

    model.load_state_dict(checkpoint, strict=False)


def to_bgr(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-p", "--pretrain_path", required=True)
    ap.add_argument("-s", "--size", type=str, default="(32, 32)")
    ap.add_argument("-c", "--min-conf", type=float, default=0.9)
    ap.add_argument("-v", "--visualize", type=int, default=-1)
    ap.add_argument("-l", "--class_labels", type=str, required=True)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--win-step", type=int, default=16)
    ap.add_argument("--pyr-scale", type=float, default=1.5)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    # -----------------------------------------------------
    # ROI Size
    # -----------------------------------------------------
    try:
        ROI_SIZE = tuple(ast.literal_eval(args.size))
        assert len(ROI_SIZE) == 2
    except Exception:
        raise ValueError("Invalid --size. Use '(32, 32)' format")

    INPUT_W, INPUT_H = ROI_SIZE

    # -----------------------------------------------------
    # Load class labels
    # -----------------------------------------------------
    with open(args.class_labels, "r") as f:
        class_names = json.load(f)

    if isinstance(class_names, dict):
        try:
            keys = sorted(class_names.keys(), key=lambda k: int(k))
            class_names = [class_names[k] for k in keys]
        except:
            # fallback
            class_names = list(class_names.values())

    # -----------------------------------------------------
    # Model Loading
    # -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] loading ViT model...")

    vit_model = MyViT(
        input_shape=(3, INPUT_H, INPUT_W),
        patch_size=8,
        n_blocks=2,
        hidden_d=128,
        n_heads=8,
        out_d=len(class_names)
    )

    safe_load_state(vit_model, args.pretrain_path, device)
    vit_model.to(device)
    vit_model.eval()

    # -----------------------------------------------------
    # Transforms (no resize here — cv2 handles resizing)
    # -----------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Only enable if used during training:
        # transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # -----------------------------------------------------
    # Load & preprocess image
    # -----------------------------------------------------
    orig_bgr = cv2.imread(args.image)
    if orig_bgr is None:
        raise ValueError(f"Could not read image: {args.image}")

    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    orig_rgb = imutils.resize(orig_rgb, width=args.width)
    (H, W) = orig_rgb.shape[:2]

    # -----------------------------------------------------
    # Sliding Windows
    # -----------------------------------------------------
    print("[INFO] building image pyramid & sliding windows...")
    pyramid = image_pyramid(orig_rgb, scale=args.pyr_scale, minSize=ROI_SIZE)

    rois = []
    locs = []

    start = time.time()
    for image in pyramid:
        scale = W / float(image.shape[1])

        for (x, y, roiOrig) in sliding_window(image, args.win_step, ROI_SIZE):
            # map box back to original resolution
            x0 = int(x * scale)
            y0 = int(y * scale)
            w_box = int(ROI_SIZE[0] * scale)
            h_box = int(ROI_SIZE[1] * scale)

            # clamp boundaries
            x0 = max(0, min(x0, W - 1))
            y0 = max(0, min(y0, H - 1))
            w_box = max(1, min(w_box, W - x0))
            h_box = max(1, min(h_box, H - y0))

            # resize ROI for ViT
            roi_resized = cv2.resize(roiOrig, (INPUT_W, INPUT_H))
            roi_tensor = transform(roi_resized)

            rois.append(roi_tensor.unsqueeze(0))
            locs.append((x0, y0, x0 + w_box, y0 + h_box))

    end = time.time()
    print(f"[INFO] generated {len(rois)} windows in {end - start:.4f}s")

    if len(rois) == 0:
        print("[INFO] no windows generated. Exiting.")
        return

    # -----------------------------------------------------
    # Batch Classification
    # -----------------------------------------------------
    all_rois = torch.cat(rois, dim=0)
    batch_size = args.batch_size
    probs_list = []

    print("[INFO] running batch inference...")
    start = time.time()

    with torch.no_grad():
        for i in range(0, len(all_rois), batch_size):
            batch = all_rois[i:i+batch_size].to(device)
            logits = vit_model(batch)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu())

    probs_all = torch.cat(probs_list, dim=0).numpy()
    end = time.time()
    print(f"[INFO] classification done in {end - start:.4f}s")

    # -----------------------------------------------------
    # Collect Detections
    # -----------------------------------------------------
    labels = {}
    for i, pvec in enumerate(probs_all):
        prob = float(pvec.max())
        label = int(pvec.argmax())
        if prob >= args.min_conf:
            labels.setdefault(label, []).append((locs[i], prob))

    if not labels:
        print("[INFO] no detections above threshold.")
        return

    # -----------------------------------------------------
    # Per-class NMS & Visualization
    # -----------------------------------------------------
    for label, boxes_probs in labels.items():
        print(f"[INFO] detections for label {label}")

        boxes = np.array([bp[0] for bp in boxes_probs]).astype("int")
        scores = np.array([bp[1] for bp in boxes_probs]).astype("float")

        pick = non_max_suppression(boxes, probs=scores)

        drawn = orig_rgb.copy()
        for (sx, sy, ex, ey) in pick:
            score = max(scores)  # approximate score (NMS strips mapping)
            label_name = class_names[label]

            cv2.rectangle(drawn, (sx, sy), (ex, ey), (0, 255, 0), 2)

            y = sy - 10 if sy - 10 > 10 else sy + 10
            cv2.putText(
                drawn,
                f"{label_name} {score:.2f}",
                (sx, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv2.imshow(f"Detections (label {label})", to_bgr(drawn))
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
