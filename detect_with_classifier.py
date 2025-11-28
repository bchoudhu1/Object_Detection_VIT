#!/usr/bin/env python3
# ---------------------------------------------------------------
# Clean + corrected sliding-window object detector for MyViT
# ---------------------------------------------------------------
# Works with any MyViT classifier trained on 32×32 images
# Includes correct scaling, batching, NMS, normalization, ROI fix,
# and proper model forward() usage.
# ---------------------------------------------------------------
#Code is still experimental. Might need more revision.

import argparse
import ast
import json
from collections import defaultdict

import cv2
import numpy as np
import torch
from torchvision import transforms

from vit_model import MyViT


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def sliding_window(image, step, ws):
    """Yield (x, y, window) for each window of size ws with step."""
    w, h = ws
    H, W = image.shape[:2]

    for y in range(0, H - h + 1, step):
        for x in range(0, W - w + 1, step):
            yield x, y, image[y:y + h, x:x + w]


def image_pyramid(image, scale=1.5, min_size=(32, 32)):
    """Yield images of decreasing size."""
    yield image
    while True:
        w = int(image.shape[1] / scale)
        if w < min_size[0]:
            break

        new = cv2.resize(image, (w, int(image.shape[0] / scale)))
        if new.shape[0] < min_size[1]:
            break

        yield new
        image = new


def safe_load_state(model, path, device):
    """Load checkpoints safely, removing module.* prefixes if needed."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    clean = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    model.load_state_dict(clean, strict=False)


def nms(boxes, scores, iou_thresh=0.3):
    """Perform Non-Max Suppression. Returns indices of boxes to keep."""
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(float)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


# ---------------------------------------------------------
# Main detector
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-p", "--pretrain_path", required=True)
    ap.add_argument("-s", "--size", type=str, default="(32, 32)")
    ap.add_argument("-c", "--min_conf", type=float, default=0.9)
    ap.add_argument("-l", "--class_labels", required=True)
    ap.add_argument("--win_step", type=int, default=16)
    ap.add_argument("--pyr_scale", type=float, default=1.5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--display_width", type=int, default=600)
    args = ap.parse_args()

    # -----------------------------------------------------
    # ROI size
    # -----------------------------------------------------
    ROI_SIZE = tuple(ast.literal_eval(args.size))
    if len(ROI_SIZE) != 2:
        raise ValueError("Use --size '(32, 32)' format.")

    W0, H0 = ROI_SIZE

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------
    with open(args.class_labels) as f:
        class_names = json.load(f)

    if isinstance(class_names, dict):
        class_names = [class_names[str(i)] for i in sorted(map(int, class_names))]

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] loading MyViT...")

    model = MyViT(
        input_shape=(3, H0, W0),
        patch_size=8,
        n_blocks=2,
        hidden_d=128,
        n_heads=8,
        out_d=len(class_names)
    )

    safe_load_state(model, args.pretrain_path, device)
    model.to(device)
    model.eval()

    # IMPORTANT: normalization matching your training pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------
    orig = cv2.imread(args.image)
    if orig is None:
        raise ValueError("Could not load image.")

    orig = cv2.resize(orig, (args.display_width,
                             int(orig.shape[0] * args.display_width / orig.shape[1])))
    H_img, W_img = orig.shape[:2]

    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # -----------------------------------------------------
    # Generate sliding windows
    # -----------------------------------------------------
    print("[INFO] generating windows...")

    rois = []
    locs = []

    for layer in image_pyramid(rgb, scale=args.pyr_scale, min_size=ROI_SIZE):
        scale = W_img / float(layer.shape[1])

        for (x, y, small_roi) in sliding_window(layer, args.win_step, ROI_SIZE):
            x0 = int(x * scale)
            y0 = int(y * scale)
            x1 = int((x + W0) * scale)
            y1 = int((y + H0) * scale)

            # ensure correct size
            if small_roi.shape[:2] != (H0, W0):
                small_roi = cv2.resize(small_roi, (W0, H0))

            rois.append(transform(small_roi).unsqueeze(0))
            locs.append((x0, y0, x1, y1))

    if len(rois) == 0:
        print("[INFO] No windows generated.")
        return

    all_rois = torch.cat(rois, dim=0)

    # -----------------------------------------------------
    # Run model inference
    # -----------------------------------------------------
    print("[INFO] running inference...")
    scores_all = []

    with torch.no_grad():
        for i in range(0, len(all_rois), args.batch_size):
            batch = all_rois[i:i + args.batch_size].to(device)

            logits = model(batch)      # FIXED
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            probs = torch.softmax(logits, dim=1)
            scores_all.append(probs.cpu())

    scores = torch.cat(scores_all, dim=0).numpy()

    # -----------------------------------------------------
    # Collect high-confidence detections
    # -----------------------------------------------------
    detections = defaultdict(list)

    for i, s in enumerate(scores):
        conf = float(s.max())
        label = int(s.argmax())

        if conf >= args.min_conf:
            detections[label].append((locs[i], conf))

    if not detections:
        print("[INFO] no detections above threshold.")
        return

    # -----------------------------------------------------
    # Apply NMS per class and display
    # -----------------------------------------------------
    for label, det_list in detections.items():
        boxes = np.array([d[0] for d in det_list])
        confs = np.array([d[1] for d in det_list])

        keep = nms(boxes, confs, iou_thresh=0.3)
        out = orig.copy()

        for idx in keep:
            (x0, y0, x1, y1) = boxes[idx]
            score = confs[idx]

            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(out,
                        f"{class_names[label]} {score:.2f}",
                        (x0, max(15, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        cv2.imshow(f"Detections: {class_names[label]}", out)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
