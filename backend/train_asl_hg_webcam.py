from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / ".vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from train_webcam_classifier import ImageArrayDataset, LABEL_MAP, LETTERS, SIZE, SmallRGBClassifier, evaluate


ARCHIVE = ROOT / "data" / "raw" / "asl-hg-raw-v1.zip"
CACHE = ROOT / "data" / "processed" / "asl_hg_signer_split"
MODEL_PATH = ROOT / "models" / "webcam_asl_rgb.pt"
EVAL_DIR = ROOT / "evaluation" / "asl_hg_webcam"
EXPECTED_SIZE = 876_640_391
EXPECTED_SHA256 = "594cfa0158044085ed61351315c187a6f3a3f9087795b4f4cdba68ecd04f12b1"
SEED = 42
EPOCHS = 10


def collect_records() -> tuple[dict[str, list[tuple[str, str, str]]], dict]:
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError("ASL-HG raw archive is missing or has the wrong file size")
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError(f"ASL-HG SHA-256 mismatch: {digest.hexdigest()}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ASL-HG archive CRC failure: {bad}")
        groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        all_classes = Counter()
        for info in archive.infolist():
            parts = info.filename.replace("\\", "/").split("/")
            if len(parts) != 3 or parts[0] != "asl_dataset" or not parts[-1].lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            label = parts[1].upper()
            stem_parts = Path(parts[-1]).stem.split("_")
            if len(stem_parts) < 3:
                raise RuntimeError(f"Cannot parse participant/sample identity from {info.filename}")
            subject, filename_label = stem_parts[0], stem_parts[1].upper()
            if filename_label != label:
                raise RuntimeError(f"Folder/filename label mismatch: {info.filename}")
            all_classes[label] += 1
            if label in LABEL_MAP:
                groups[subject].append((info.filename, label, subject))
        if set(all_classes) != set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
            raise RuntimeError(f"Unexpected ASL-HG classes: {sorted(all_classes)}")
        if any(count != 1000 for count in all_classes.values()):
            raise RuntimeError(f"Expected 1,000 images per ASL-HG class: {dict(all_classes)}")
        subjects = sorted(groups)
        if len(subjects) != 10:
            raise RuntimeError(f"Expected 10 volunteer IDs, found {subjects}")
        for subject in subjects:
            counts = Counter(label for _, label, _ in groups[subject])
            if counts != Counter({letter: 100 for letter in LETTERS}):
                raise RuntimeError(f"Unexpected supported class counts for {subject}: {dict(counts)}")
        # Read representative JPEGs from every signer and supported class before fitting.
        formats = Counter()
        sample_count = 0
        for subject in subjects:
            by_class = {label: name for name, label, _ in groups[subject]}
            for filename in by_class.values():
                with archive.open(filename) as file, Image.open(file) as image:
                    image.verify()
                with archive.open(filename) as file, Image.open(file) as image:
                    formats[f"{image.format}:{image.width}x{image.height}:{image.mode}"] += 1
                sample_count += 1
        verification = {
            "status": "passed",
            "source": "Mendeley Data ASL-HG, version 1",
            "license": "CC BY 4.0",
            "archive": ARCHIVE.name,
            "bytes": ARCHIVE.stat().st_size,
            "sha256": EXPECTED_SHA256,
            "sha256_check": "passed",
            "zip_crc": "passed",
            "all_class_counts": dict(sorted(all_classes.items())),
            "supported_letters": LETTERS,
            "supported_rows": sum(sum(1 for _, label, _ in rows) for rows in groups.values()),
            "volunteers": {subject: len(groups[subject]) for subject in subjects},
            "representative_images_decoded": sample_count,
            "representative_format_dimensions_modes": dict(formats),
            "preprocessing": f"RGB conversion, {SIZE}x{SIZE} Lanczos resize, uint8 cache; per-signer train/validation/test split.",
        }
        return groups, verification


def crop_like_camera(image: Image.Image, landmarks):
    width, height = image.size
    xs = [point.x * width for point in landmarks]
    ys = [point.y * height for point in landmarks]
    box_width = max(xs) - min(xs)
    box_height = max(ys) - min(ys)
    side = max(box_width, box_height) * 1.5
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    side = max(1.0, min(side, width, height))
    left = max(0.0, min(width - side, center_x - side / 2))
    top = max(0.0, min(height - side, center_y - side / 2))
    return image.crop((left, top, left + side, top + side))


def materialize(archive: zipfile.ZipFile, landmarker, rows, name: str):
    CACHE.mkdir(parents=True, exist_ok=True)
    pixels_path = CACHE / f"{name}_pixels.npy"
    labels_path = CACHE / f"{name}_labels.npy"
    pixels = np.lib.format.open_memmap(pixels_path, mode="w+", dtype=np.uint8, shape=(len(rows), SIZE, SIZE, 3))
    labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.uint8, shape=(len(rows),))
    counts = Counter()
    failed_hand_detection = []
    output_index = 0
    for index, (filename, label, subject) in enumerate(rows):
        with archive.open(filename) as file, Image.open(file) as image:
            image = image.convert("RGB")
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image, dtype=np.uint8))
            detected = landmarker.detect(mp_image).hand_landmarks
            if not detected:
                failed_hand_detection.append(filename)
                continue
            crop = crop_like_camera(image, detected[0])
            pixels[output_index] = np.asarray(crop.resize((SIZE, SIZE), Image.Resampling.LANCZOS), dtype=np.uint8)
        labels[output_index] = LABEL_MAP[label]
        output_index += 1
        counts[label] += 1
        if (index + 1) % 2000 == 0 or index + 1 == len(rows):
            print(f"Preprocessed {name}: {index + 1}/{len(rows)} checked; {output_index} hand crops retained", flush=True)
    if set(counts) != set(LETTERS) or any(count == 0 for count in counts.values()):
        raise RuntimeError(f"MediaPipe crop filtering left a missing supported class in {name}: {dict(counts)}")
    pixels.flush()
    labels.flush()
    (CACHE / f"{name}_manifest.json").write_text(json.dumps({"split": name, "candidate_rows": len(rows), "retained_rows": output_index, "skipped_no_hand_detected": len(failed_hand_detection), "skipped_examples": failed_hand_detection[:50], "class_counts": dict(counts), "size": [SIZE, SIZE], "channels": 3, "preprocessing": "MediaPipe 21-landmark crop, square crop padded by 1.5x hand-box size to match the browser input, RGB Lanczos resize"}, indent=2), encoding="utf-8")
    return np.load(pixels_path, mmap_mode="r")[:output_index], np.load(labels_path, mmap_mode="r")[:output_index]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    grouped, verification = collect_records()
    people = sorted(grouped)
    random.Random(SEED).shuffle(people)
    train_people, validation_people, test_people = people[:8], people[8:9], people[9:10]
    split = {
        "train": [row for person in train_people for row in grouped[person]],
        "validation": [row for person in validation_people for row in grouped[person]],
        "test": [row for person in test_people for row in grouped[person]],
    }
    verification["signer_split"] = {"seed": SEED, "train_volunteers": train_people, "validation_volunteers": validation_people, "test_volunteers": test_people, "overlap": False}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "dataset_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(f"Verified ASL-HG: {verification['supported_rows']} supported images; 8 train signers, 1 validation signer, 1 untouched test signer; 24 classes.", flush=True)

    landmarker_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ROOT.parent / "frontend" / "public" / "hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.4,
    )
    with zipfile.ZipFile(ARCHIVE) as archive, vision.HandLandmarker.create_from_options(landmarker_options) as landmarker:
        train_x, train_y = materialize(archive, landmarker, split["train"], "train")
        validation_x, validation_y = materialize(archive, landmarker, split["validation"], "validation")
        test_x, test_y = materialize(archive, landmarker, split["test"], "test")

    loaders = {
        "train": DataLoader(ImageArrayDataset(train_x, train_y, True), batch_size=128, shuffle=True, num_workers=0),
        "validation": DataLoader(ImageArrayDataset(validation_x, validation_y, False), batch_size=256, shuffle=False, num_workers=0),
        "test": DataLoader(ImageArrayDataset(test_x, test_y, False), batch_size=256, shuffle=False, num_workers=0),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallRGBClassifier(len(LETTERS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    history = []
    best_accuracy = -1.0
    best_state = None
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        correct = total = 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            total += len(y)
        scheduler.step()
        validation = evaluate(model, loaders["validation"], device)
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total, "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"]}
        history.append(row)
        print("Epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f} val_macro_f1={validation_macro_f1:.4f}".format(epochs=EPOCHS, **row), flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training produced no validation-selected checkpoint")
    model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], device)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "label_map": LABEL_MAP, "num_classes": len(LETTERS), "input_size": SIZE, "input_channels": 3, "seed": SEED, "dataset": "Mendeley ASL-HG v1", "test_volunteer": test_people[0]}, MODEL_PATH)
    (EVAL_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "task": "static ASL fingerspelling, 24 classes A-I and K-Y",
        "dataset": "Mendeley ASL-HG v1 (CC BY 4.0)",
        "source_rows": 24000,
        "train_rows": len(train_y), "validation_rows": len(validation_y), "test_rows": len(test_y),
        "train_volunteers": train_people, "validation_volunteer": validation_people[0], "test_volunteer": test_people[0],
        "hand_detection_coverage": {name: len(y) / len(split[name]) for name, y in (("train", train_y), ("validation", validation_y), ("test", test_y))},
        "signer_independent_split": True,
        "seed": SEED, "epochs": EPOCHS, "image_size": [SIZE, SIZE], "channels": 3,
        "crop": "MediaPipe 21-landmark square crop, 1.5x padding (same as frontend)",
        "best_validation_accuracy": best_accuracy,
        "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"], "test_correct": test["correct"],
        "device": str(device), "training_seconds": round(time.time() - started, 2), "checkpoint": str(MODEL_PATH.relative_to(ROOT)),
        "note": "Test signer was excluded from training, validation, and checkpoint selection. A prior full-frame baseline was evaluated on this volunteer during development, so this is not a pristine blind challenge. This single-signer result is not a guarantee of accuracy for arbitrary webcam users.",
    }
    (EVAL_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savetxt(EVAL_DIR / "confusion_matrix.csv", test["confusion"], delimiter=",", fmt="%d", header=",".join(LETTERS), comments="")
    with (EVAL_DIR / "classification_report.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["letter", "correct", "total", "recall"])
        for index, letter in enumerate(LETTERS):
            count = int(test["confusion"][index].sum())
            writer.writerow([letter, int(test["confusion"][index, index]), count, f"{test['confusion'][index,index]/count:.6f}"])
    print("TEST RESULTS", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
