from __future__ import annotations

import csv
import json
import random
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "webcam_asl_rgb"
MODEL_PATH = ROOT / "models" / "webcam_asl_rgb.pt"
EVAL_DIR = ROOT / "evaluation" / "webcam_asl_rgb"
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
LABEL_MAP = {letter: index for index, letter in enumerate(LETTERS)}
SIZE = 64
SEED = 42
EPOCHS = 10
ImageFile.LOAD_TRUNCATED_IMAGES = False


def image_name(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


def archive_records(path: Path, source: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with zipfile.ZipFile(path) as archive:
        if source == "grassknoted":
            for info in archive.infolist():
                parts = info.filename.replace("\\", "/").split("/")
                if "asl_alphabet_train" in parts and len(parts) >= 2 and image_name(parts[-1]):
                    label = parts[-2].upper()
                    if label in LABEL_MAP:
                        records.append((info.filename, label))
        elif source == "ayuraj":
            for info in archive.infolist():
                parts = info.filename.replace("\\", "/").split("/")
                if len(parts) == 3 and parts[0] == "asl_dataset" and image_name(parts[-1]):
                    label = parts[1].upper()
                    if label in LABEL_MAP:
                        records.append((info.filename, label))
        elif source == "danrasband":
            for info in archive.infolist():
                parts = info.filename.replace("\\", "/").split("/")
                if len(parts) == 2 and image_name(parts[-1]):
                    label = parts[0].upper()
                    if label in LABEL_MAP:
                        records.append((info.filename, label))
    return records


def verify_records(records: list[tuple[str, str]], source: str, expected: int) -> None:
    counts = Counter(label for _, label in records)
    if len(records) != expected or set(counts) != set(LETTERS):
        raise RuntimeError(f"{source}: unexpected 24-class records ({len(records)}); counts={dict(counts)}")
    expected_per_class = expected // len(LETTERS)
    if source == "ayuraj":
        if any(n not in (65, 70) for n in counts.values()):
            raise RuntimeError(f"{source}: unexpected class counts {dict(counts)}")
    elif any(n != expected_per_class for n in counts.values()):
        raise RuntimeError(f"{source}: unexpected class counts {dict(counts)}")


class SmallRGBClassifier(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(256, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ImageArrayDataset(Dataset):
    def __init__(self, pixels: np.ndarray, labels: np.ndarray, augment: bool):
        self.pixels = pixels
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.pixels[index].copy()).permute(2, 0, 1).float().div_(255)
        if self.augment:
            if random.random() < 0.5:
                x = x.flip(-1)
            brightness = random.uniform(0.78, 1.22)
            contrast = random.uniform(0.78, 1.22)
            mean = x.mean(dim=(1, 2), keepdim=True)
            x = ((x - mean) * contrast + mean) * brightness
            x.clamp_(0, 1)
        return x, int(self.labels[index])


def stratified_split(records: list[tuple[str, str]], validation_fraction: float, rng: random.Random):
    grouped: dict[str, list[tuple[str, str]]] = {letter: [] for letter in LETTERS}
    for record in records:
        grouped[record[1]].append(record)
    train, validation = [], []
    for label in LETTERS:
        items = grouped[label]
        rng.shuffle(items)
        n_validation = max(1, round(len(items) * validation_fraction))
        validation.extend(items[:n_validation])
        train.extend(items[n_validation:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def materialize(records_by_source: dict[str, list[tuple[str, str]]], split_name: str) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[str, str, str]] = []
    for source, records in records_by_source.items():
        rows.extend((source, name, label) for name, label in records)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pixels_path = PROCESSED / f"{split_name}_pixels.npy"
    labels_path = PROCESSED / f"{split_name}_labels.npy"
    pixels = np.lib.format.open_memmap(pixels_path, mode="w+", dtype=np.uint8, shape=(len(rows), SIZE, SIZE, 3))
    labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.uint8, shape=(len(rows),))
    open_archives: dict[str, zipfile.ZipFile] = {}
    try:
        for index, (source, filename, label) in enumerate(rows):
            archive = open_archives.get(source)
            if archive is None:
                archive_name = {"grassknoted": "grassknoted-asl-alphabet-v1.zip", "ayuraj": "ayuraj-asl-dataset-v1.zip", "danrasband": "danrasband-asl-alphabet-test-v1.zip"}[source]
                archive = zipfile.ZipFile(RAW / archive_name)
                open_archives[source] = archive
            with archive.open(filename) as file, Image.open(file) as image:
                image = image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
                pixels[index] = np.asarray(image, dtype=np.uint8)
            labels[index] = LABEL_MAP[label]
            if (index + 1) % 10000 == 0 or index + 1 == len(rows):
                print(f"Preprocessed {split_name}: {index + 1}/{len(rows)}", flush=True)
    finally:
        for archive in open_archives.values():
            archive.close()
    pixels.flush()
    labels.flush()
    manifest = {"split": split_name, "rows": len(rows), "sources": dict(Counter(source for source, _, _ in rows)), "class_counts": dict(Counter(label for _, _, label in rows)), "size": [SIZE, SIZE], "channels": 3}
    (PROCESSED / f"{split_name}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return np.load(pixels_path, mmap_mode="r"), np.load(labels_path, mmap_mode="r")


def evaluate(model, loader, device):
    model.eval()
    confusion = np.zeros((len(LETTERS), len(LETTERS)), dtype=np.int64)
    loss_sum = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(criterion(logits, y).item())
            pred = logits.argmax(1)
            for truth, guessed in zip(y.cpu().numpy(), pred.cpu().numpy()):
                confusion[int(truth), int(guessed)] += 1
            total += len(y)
    correct = int(np.trace(confusion))
    true_positive = np.diag(confusion).astype(np.float64)
    false_positive = confusion.sum(0) - true_positive
    false_negative = confusion.sum(1) - true_positive
    denominators = 2 * true_positive + false_positive + false_negative
    per_class_f1 = np.divide(2 * true_positive, denominators, out=np.zeros(len(LETTERS)), where=denominators > 0)
    return {"loss": loss_sum / max(1, total), "accuracy": correct / max(1, total), "macro_f1": float(per_class_f1.mean()), "correct": correct, "total": total, "confusion": confusion}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    archive_names = {"grassknoted": "grassknoted-asl-alphabet-v1.zip", "ayuraj": "ayuraj-asl-dataset-v1.zip", "danrasband": "danrasband-asl-alphabet-test-v1.zip"}
    integrity = {}
    for source, filename in archive_names.items():
        path = RAW / filename
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Dataset archive failed CRC verification: {filename}: {bad_member}")
            integrity[source] = {"archive": filename, "bytes": path.stat().st_size, "entries": len(archive.infolist()), "crc_check": "passed"}
    sources = {
        "grassknoted": archive_records(RAW / "grassknoted-asl-alphabet-v1.zip", "grassknoted"),
        "ayuraj": archive_records(RAW / "ayuraj-asl-dataset-v1.zip", "ayuraj"),
        "danrasband": archive_records(RAW / "danrasband-asl-alphabet-test-v1.zip", "danrasband"),
    }
    verify_records(sources["grassknoted"], "grassknoted", 72000)
    verify_records(sources["ayuraj"], "ayuraj", 1675)
    verify_records(sources["danrasband"], "danrasband", 720)
    for source in sources:
        archive = zipfile.ZipFile(RAW / archive_names[source])
        try:
            sample_by_class: dict[str, str] = {}
            for filename, label in sources[source]:
                sample_by_class.setdefault(label, filename)
            sizes = set()
            for filename in sample_by_class.values():
                with archive.open(filename) as file, Image.open(file) as image:
                    image.verify()
                with archive.open(filename) as file, Image.open(file) as image:
                    sizes.add(f"{image.format}:{image.width}x{image.height}:{image.mode}")
            integrity[source]["sample_image_formats_dimensions_mode"] = sorted(sizes)
            integrity[source]["verified_class_count"] = len(sample_by_class)
        finally:
            archive.close()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "dataset_verification.json").write_text(json.dumps({"status": "passed", "classes": LETTERS, "archives": integrity, "records_by_source": {source: len(records) for source, records in sources.items()}}, indent=2), encoding="utf-8")
    print("Dataset preflight passed: 72,000 primary train candidates, 1,675 supplemental samples, 720 independent test samples; all 24 supported classes present.", flush=True)

    rng = random.Random(SEED)
    main_train, main_val = stratified_split(sources["grassknoted"], 0.10, rng)
    aux_train, aux_val = stratified_split(sources["ayuraj"], 0.20, rng)
    train_x, train_y = materialize({"grassknoted": main_train, "ayuraj": aux_train}, "train")
    val_x, val_y = materialize({"grassknoted": main_val, "ayuraj": aux_val}, "validation")
    test_x, test_y = materialize({"danrasband": sources["danrasband"]}, "test")

    train_dataset = ImageArrayDataset(train_x, train_y, augment=True)
    val_dataset = ImageArrayDataset(val_x, val_y, augment=False)
    test_dataset = ImageArrayDataset(test_x, test_y, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallRGBClassifier(len(LETTERS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    history = []
    best_val = -1.0
    best_state = None
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        correct = total = 0
        for x, y in train_loader:
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
        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / max(1, total), "train_accuracy": correct / max(1, total), "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"]}
        history.append(row)
        print("Epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f} val_macro_f1={validation_macro_f1:.4f}".format(epochs=EPOCHS, **row), flush=True)
        if validation["accuracy"] > best_val:
            best_val = validation["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No checkpoint selected from validation")
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "label_map": LABEL_MAP, "num_classes": len(LETTERS), "input_size": SIZE, "input_channels": 3, "seed": SEED, "source": "public labeled ASL alphabet datasets; see README"}, MODEL_PATH)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "task": "static ASL fingerspelling, 24 classes A-I and K-Y",
        "dataset": {"grassknoted_training_candidates": 72000, "ayuraj_supplemental_candidates": 1675, "danrasband_independent_test_candidates": 720, "train_rows": int(len(train_y)), "validation_rows": int(len(val_y)), "test_rows": int(len(test_y)), "seed": SEED, "image_size": [SIZE, SIZE], "channels": 3, "excluded_classes": ["J", "Z", "SPACE", "DELETE", "NOTHING", "digits"]},
        "best_validation_accuracy": best_val,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_correct": test["correct"],
        "test_rows": test["total"],
        "device": str(device),
        "training_seconds": round(time.time() - start, 2),
        "checkpoint": str(MODEL_PATH.relative_to(ROOT)),
        "note": "Independent Dan Rasband test set was not used for training or model selection. Scores measure that public image set, not personalized webcam accuracy.",
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
