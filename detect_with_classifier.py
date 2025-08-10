#!/usr/bin/env python3
# refactored detector using ViT, batching, correct softmax, and safer parsing.
#This code is under progress, is highly experimental and might contain bugs.
#Will need further review.
#Adapted from https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/
#!/usr/bin/env python3
# refactored detector using ViT, batching, correct softmax, and safer parsing.
# import the necessary packages
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
from vit_model import MyViT  # import from ViT implementation


def sliding_window(image, step, ws):
    # ws = (w, h)
    # include final positions with +1 to avoid off-by-one
    for y in range(0, max(1, image.shape[0] - ws[1] + 1), step):
        for x in range(0, max(1, image.shape[1] - ws[0] + 1), step):
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])


def image_pyramid(image, scale=1.5, minSize=(32, 32)):
    # yield the original image, then smaller ones
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


def safe_load_state(model, path, device):
    # load with map_location, handle 'module.' prefix if present
    checkpoint = torch.load(path, map_location=device)
    # if saved as dict with 'state_dict', unwrap
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    # Fix keys if DataParallel used
    if any(k.startswith("module.") for k in checkpoint.keys()):
        new_state = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "")
            new_state[name] = v
        checkpoint = new_state
    model.load_state_dict(checkpoint)


def to_bgr(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    ap.add_argument("-p", "--pretrain_path", required=True, help="path to pretrained model")
    ap.add_argument("-s", "--size", type=str, default="(32, 32)", help="ROI size as tuple string, e.g. '(32, 32)'")
    ap.add_argument("-c", "--min-conf", type=float, default=0.9, help="minimum probability to keep detection")
    ap.add_argument("-v", "--visualize", type=int, default=-1, help="show debug windows")
    ap.add_argument("-l", "--class_labels", type=str, required=True, help="JSON file mapping classes (list or dict)")
    ap.add_argument("--width", type=int, default=400, help="resize image width")
    ap.add_argument("--win-step", type=int, default=16)
    ap.add_argument("--pyr-scale", type=float, default=1.5)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    # safe parse size
    try:
        ROI_SIZE = tuple(ast.literal_eval(args.size))
        assert len(ROI_SIZE) == 2
    except Exception:
        raise ValueError("Invalid --size. Use a tuple like '(32, 32)'")

    INPUT_SIZE = (32, 32)  # (W, H) used for cv2.resize; transforms.Resize uses (H, W)
    WIDTH = args.width
    PYR_SCALE = args.pyr_scale
    WIN_STEP = args.win_step

    # load class names
    with open(args.class_labels, "r") as f:
        class_names = json.load(f)
    # normalize class_names to a list when possible
    if isinstance(class_names, dict):
        # try to convert dict keys "0","1",... to list by sorted keys
        try:
            keys = sorted(class_names.keys(), key=lambda k: int(k))
            class_list = [class_names[k] for k in keys]
            class_names = class_list
        except Exception:
            # keep dict as-is; we'll do str lookup later
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] loading Vision Transformer model...")
    # adjust constructor args to match your MyViT
    vit_model = MyViT(chw=(3, 32, 32), n_patches=7, n_blocks=2, hidden_d=128, n_heads=8, out_d=10)
    safe_load_state(vit_model, args.pretrain_path, device)
    vit_model.to(device)
    vit_model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((INPUT_SIZE[1], INPUT_SIZE[0])),  # PIL expects (H,W)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    orig_bgr = cv2.imread(args.image)
    if orig_bgr is None:
        raise ValueError(f"Could not read image: {args.image}")
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    orig_rgb = imutils.resize(orig_rgb, width=WIDTH)
    (H, W) = orig_rgb.shape[:2]

    print("[INFO] building image pyramid and sliding windows...")
    pyramid = image_pyramid(orig_rgb, scale=PYR_SCALE, minSize=ROI_SIZE)

    rois = []
    locs = []
    start = time.time()
    for image in pyramid:
        scale = W / float(image.shape[1])
        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            # map window coords to original resized image coords
            x0 = int(x * scale)
            y0 = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            # clamp
            x0 = max(0, min(x0, W - 1))
            y0 = max(0, min(y0, H - 1))
            w = max(1, min(w, W - x0))
            h = max(1, min(h, H - y0))

            # prepare tensor on CPU (do NOT move to device yet)
            roi_resized = cv2.resize(roiOrig, INPUT_SIZE)  # (W,H)
            roi_tensor = transform(roi_resized)  # C,H,W
            rois.append(roi_tensor.unsqueeze(0))  # 1,C,H,W on CPU
            locs.append((x0, y0, x0 + w, y0 + h))

            if args.visualize > 0:
                vis = orig_rgb.copy()
                cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 1)
                cv2.imshow("Window (BGR)", to_bgr(vis))
                cv2.imshow("ROI (BGR)", to_bgr(roiOrig))
                cv2.waitKey(1)

    end = time.time()
    print(f"[INFO] pyramid/windows loop took {end - start:.4f} seconds. {len(rois)} windows.")

    if len(rois) == 0:
        print("[INFO] no candidate windows produced. Exiting.")
        return

    # batch inference with softmax -> probabilities
    all_rois = torch.cat(rois, dim=0)
    batch_size = args.batch_size
    probs_list = []
    start = time.time()
    with torch.no_grad():
        for i in range(0, all_rois.size(0), batch_size):
            batch = all_rois[i:i + batch_size].to(device)
            logits = vit_model(batch)  # shape (B, num_classes)
            probs = torch.softmax(logits, dim=1)  # now true probabilities
            probs_list.append(probs.cpu())
    probs_all = torch.cat(probs_list, dim=0).numpy()
    end = time.time()
    print(f"[INFO] classifying ROIs took {end - start:.4f} seconds")

    # build label -> [(box, prob), ...]
    labels = {}
    for i, pvec in enumerate(probs_all):
        prob = float(pvec.max())
        label = int(pvec.argmax())
        if prob >= args.min_conf:
            labels.setdefault(label, []).append((locs[i], prob))

    if not labels:
        print("[INFO] no detections above min confidence.")
        return

    # visualize per-class with NMS
    for label, boxes_probs in labels.items():
        print(f"[INFO] showing results for label {label}")
        before = orig_rgb.copy()
        for (box, prob) in boxes_probs:
            (sx, sy, ex, ey) = box
            cv2.rectangle(before, (sx, sy), (ex, ey), (0, 255, 0), 1)
        cv2.imshow("Before NMS", to_bgr(before))

        boxes = np.array([bp[0] for bp in boxes_probs]).astype("int")
        scores = np.array([bp[1] for bp in boxes_probs]).astype("float")
        # imutils' NMS expects boxes and scores; returns kept boxes
        pick = non_max_suppression(boxes, probs=scores) if scores.size else np.array([])

        after = orig_rgb.copy()
        for (sx, sy, ex, ey) in pick:
            cv2.rectangle(after, (sx, sy), (ex, ey), (0, 255, 0), 2)
            # safe label name lookup
            if isinstance(class_names, list) and label < len(class_names):
                label_name = class_names[label]
            elif isinstance(class_names, dict):
                label_name = class_names.get(str(label), class_names.get(label, f"label_{label}"))
            else:
                label_name = f"label_{label}"
            y = sy - 10 if sy - 10 > 10 else sy + 10
            cv2.putText(after, f"{label_name} {scores.tolist()[0]:.2f}", (sx, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.imshow("After NMS", to_bgr(after))
        cv2.waitKey(0)

if __name__ == "__main__":
    main()

