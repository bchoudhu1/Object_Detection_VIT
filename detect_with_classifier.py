#!/usr/bin/env python3
# Fully corrected sliding-window + pyramid detector for MyViT
# Compatible with ViT input_shape = (3, 32, 32)
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

from vit_model import MyViT  # your ViT network


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def sliding_window(image, step, ws):
    (w, h) = ws
    H, W = image.shape[:2]

    for y in range(0, H - h + 1, step):
        for x in range(0, W - w + 1, step):
            yield (x, y, image[y:y + h, x:x + w])


def image_pyramid(image, scale=1.5, minSize=(32, 32)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        if w < minSize[0]:
            break
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1]:
            break
        yield image


def safe_load_state(model, path, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # Remove "module." prefix if present
    cleaned = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)


def to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-p", "--pretrain_path", required=True)
    ap.add_argument("-s", "--size", type=str, default="(32, 32)")
    ap.add_argument("-c", "--min-conf", type=float, default=0.9)
    ap.add_argument("-l", "--class_labels", type=str, required=True)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--win-step", type=int, default=16)
    ap.add_argument("--pyr-scale", type=float, default=1.5)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    # ROI size
    ROI_SIZE = tuple(ast.literal_eval(args.size))
    if len(ROI_SIZE) != 2:
        raise ValueError("Use size format like '(32, 32)'")

    INPUT_W, INPUT_H = ROI_SIZE

    # Load label names
    with open(args.class_labels, "r") as f:
        class_names = json.load(f)

    if isinstance(class_names, dict):
        keys = sorted(class_names.keys(), key=lambda k: int(k))
        class_names = [class_names[k] for k in keys]

    # Device + model
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

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load image
    orig_bgr = cv2.imread(args.image)
    if orig_bgr is None:
        raise ValueError(f"Could not load image {args.image}")

    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    orig_rgb = imutils.resize(orig_rgb, width=args.width)
    (H, W) = orig_rgb.shape[:2]

    # Sliding windows
    print("[INFO] creating pyramid & windows...")
    rois = []
    locs = []

    pyramid = list(image_pyramid(orig_rgb, args.pyr_scale, ROI_SIZE))

    start = time.time()
    for layer in pyramid:
        scale = W / float(layer.shape[1])

        for (x, y, roiOrig) in sliding_window(layer, args.win_step, ROI_SIZE):
            # map to original coordinates
            x0 = int(x * scale)
            y0 = int(y * scale)
            w_box = int(INPUT_W * scale)
            h_box = int(INPUT_H * scale)

            # clamp
            x1 = min(W - 1, x0 + w_box)
            y1 = min(H - 1, y0 + h_box)

            # Resize ROI to ViT size
            roi_resized = cv2.resize(roiOrig, (INPUT_W, INPUT_H))
            roi_tensor = transform(roi_resized).unsqueeze(0)

            rois.append(roi_tensor)
            locs.append((x0, y0, x1, y1))

    end = time.time()
    print(f"[INFO] {len(rois)} windows generated in {end - start:.4f}s")

    if len(rois) == 0:
        print("[INFO] no windows generated.")
        return

    # Batch inference
    print("[INFO] running inference...")
    start = time.time()

    all_rois = torch.cat(rois, dim=0)
    probs_list = []

    with torch.no_grad():
        for i in range(0, len(all_rois), args.batch_size):
            batch = all_rois[i:i + args.batch_size].to(device)

            logits = vit_model(batch)[0]  # FIX: ViT returns (logits,)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu())

    probs_all = torch.cat(probs_list, dim=0).numpy()

    end = time.time()
    print(f"[INFO] inference done in {end - start:.4f}s")

    # Collect predictions
    detections = {}
    for i, p in enumerate(probs_all):
        prob = float(p.max())
        label = int(p.argmax())

        if prob >= args.min_conf:
            detections.setdefault(label, []).append((locs[i], prob))

    if not detections:
        print("[INFO] No detections above threshold.")
        return

    # Draw results
    for label, box_infos in detections.items():
        print(f"[INFO] applying NMS for class: {class_names[label]}")

        boxes = np.array([b for (b, _) in box_infos])
        scores = np.array([s for (_, s) in box_infos])

        picks = non_max_suppression(boxes, probs=scores)
        drawn = orig_rgb.copy()

        for box in picks:
            (sx, sy, ex, ey) = box
            score = float(max(scores))  # approximate, NMS loses mapping

            cv2.rectangle(drawn, (sx, sy), (ex, ey), (0, 255, 0), 2)
            ytxt = sy - 10 if sy > 20 else sy + 20

            cv2.putText(
                drawn,
                f"{class_names[label]} {score:.2f}",
                (sx, ytxt),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(f"Detections: {class_names[label]}", to_bgr(drawn))
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
