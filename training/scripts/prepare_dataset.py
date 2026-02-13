"""
Split the raw COCO dataset into train/val/test, then convert train+val to YOLO format.

Pipeline:
  1. Load COCO annotations
  2. Stratified 3-way split on RAW image IDs (70/15/15 by default)
  3. Write test split as COCO JSON (evaluator consumes COCO natively)
  4. Convert train + val splits to YOLO format for Ultralytics training

The test set is never touched during training or checkpoint selection.
Final performance is reported exclusively on test.
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


TRAINING_ROOT = Path(__file__).parent.parent.resolve()
RAW_DATA = TRAINING_ROOT / "data" / "raw"
PROCESSED_DATA = TRAINING_ROOT / "data" / "processed"

DEFAULT_ANNOTATIONS = RAW_DATA / "annotations" / "_annotations.coco.json"
DEFAULT_IMAGES_DIR = RAW_DATA / "coin-dataset"


def load_coco(annotations_path: Path) -> dict:
    with open(annotations_path) as f:
        return json.load(f)


def coco_to_yolo_bbox(bbox: list, img_w: int, img_h: int) -> tuple:
    """Convert COCO bbox [x, y, w, h] to YOLO format [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def build_image_annotations(coco: dict) -> dict:
    """Group annotations by image_id."""
    image_anns = defaultdict(list)
    for ann in coco["annotations"]:
        image_anns[ann["image_id"]].append(ann)
    return image_anns


def _build_empirical_bins(
    image_ids: list,
    image_anns: dict,
    min_bin_size: int = 3,
) -> dict:
    """
    Bin images by annotation count using data-driven thresholds.

    Each annotation count that appears in >= min_bin_size images becomes
    its own stratum. Sparse counts (< min_bin_size images) are merged
    into a single tail bin keyed as "tail".

    This adapts to whatever distribution the dataset has rather than
    imposing fixed thresholds that may not reflect the data.
    """
    from collections import Counter

    # Count how many images have each annotation count
    count_per_image = {}
    for img_id in image_ids:
        count_per_image[img_id] = len(image_anns.get(img_id, []))

    count_freq = Counter(count_per_image.values())

    # Determine which counts get their own bin vs. get merged
    standalone = set()
    for count_val, freq in count_freq.items():
        if freq >= min_bin_size:
            standalone.add(count_val)

    # Assign images to bins
    bins = defaultdict(list)
    for img_id in image_ids:
        c = count_per_image[img_id]
        bin_key = c if c in standalone else "tail"
        bins[bin_key].append(img_id)

    return dict(bins)


def stratified_three_way_split(
    image_ids: list,
    image_anns: dict,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_bin_size: int = 3,
) -> tuple:
    """
    Split image IDs into train/val/test with stratification by annotation count.

    Bins are derived from the empirical distribution: each annotation count
    with >= min_bin_size images becomes its own stratum; sparse counts are
    merged into a tail bin. This preserves representativeness without
    imposing fixed thresholds.

    Within each bin, test is carved out first, then val, to ensure
    test data is never leaked into training or checkpoint selection.

    Returns:
        (train_ids, val_ids, test_ids)
    """
    rng = np.random.default_rng(seed)

    bins = _build_empirical_bins(image_ids, image_anns, min_bin_size)

    # Log bin composition
    print("  Stratification bins (empirical, min_bin_size=%d):" % min_bin_size)
    for key in sorted(bins.keys(), key=lambda k: (isinstance(k, str), k)):
        ids = bins[key]
        label = f"count={key}" if isinstance(key, int) else f"{key} (merged sparse counts)"
        print(f"    {label}: {len(ids)} images")

    train_ids, val_ids, test_ids = [], [], []

    for bin_key in sorted(bins.keys(), key=lambda k: (isinstance(k, str), k)):
        ids = list(bins[bin_key])
        rng.shuffle(ids)

        n_test = max(1, int(len(ids) * test_ratio))
        n_val = max(1, int(len(ids) * val_ratio))

        # Carve test first, then val, rest is train
        test_ids.extend(ids[:n_test])
        val_ids.extend(ids[n_test : n_test + n_val])
        train_ids.extend(ids[n_test + n_val :])

    return train_ids, val_ids, test_ids


def build_coco_subset(coco: dict, image_ids: set, id_to_image: dict, image_anns: dict) -> dict:
    """
    Build a COCO-format dict containing only the specified image IDs.

    Preserves the original annotation IDs and category definitions.
    """
    images = [id_to_image[img_id] for img_id in sorted(image_ids)]
    annotations = []
    for img_id in image_ids:
        annotations.extend(image_anns.get(img_id, []))

    return {
        "images": images,
        "annotations": sorted(annotations, key=lambda a: a["id"]),
        "categories": coco["categories"],
    }


def prepare_dataset(
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    output_dir: Path = PROCESSED_DATA,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Main pipeline: load COCO, 3-way split on raw, write test as COCO,
    convert train+val to YOLO.
    """
    print(f"Loading annotations from {annotations_path}")
    coco = load_coco(annotations_path)

    # Build lookups
    id_to_image = {img["id"]: img for img in coco["images"]}
    image_anns = build_image_annotations(coco)

    class_id = 0  # single class: coin

    image_ids = list(id_to_image.keys())
    print(f"Total images: {len(image_ids)}")
    print(f"Total annotations: {len(coco['annotations'])}")

    # ── 3-way stratified split ────────────────────────────────────────────────
    train_ids, val_ids, test_ids = stratified_three_way_split(
        image_ids, image_anns, val_ratio, test_ratio, seed
    )
    print(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Sanity: no overlap
    assert not (set(train_ids) & set(val_ids)), "train/val overlap!"
    assert not (set(train_ids) & set(test_ids)), "train/test overlap!"
    assert not (set(val_ids) & set(test_ids)), "val/test overlap!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(image_ids), "split size mismatch!"

    # ── Write test split as COCO JSON ─────────────────────────────────────────
    # The evaluator consumes COCO format natively, so test stays in COCO
    test_dir = output_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "images").mkdir(exist_ok=True)

    test_coco = build_coco_subset(coco, set(test_ids), id_to_image, image_anns)
    test_annotations_path = test_dir / "_annotations.coco.json"
    with open(test_annotations_path, "w") as f:
        json.dump(test_coco, f, indent=2)

    # Copy test images
    test_skipped = 0
    for img_id in test_ids:
        img_info = id_to_image[img_id]
        src = images_dir / img_info["file_name"]
        if not src.exists():
            print(f"  WARNING: Missing test image {img_info['file_name']}, skipping")
            test_skipped += 1
            continue
        shutil.copy2(src, test_dir / "images" / img_info["file_name"])

    print(f"Test set: {len(test_ids) - test_skipped} images → {test_dir}")
    print(f"  COCO annotations: {test_annotations_path}")

    # ── Convert train + val to YOLO format ────────────────────────────────────
    yolo_dir = output_dir / "yolo"
    for split in ("train", "val"):
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    splits_info = {
        "train": [],
        "val": [],
        "test": [],
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    skipped = 0

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in ids:
            img_info = id_to_image[img_id]
            src_path = images_dir / img_info["file_name"]

            if not src_path.exists():
                print(f"  WARNING: Missing image {img_info['file_name']}, skipping")
                skipped += 1
                continue

            # Copy image to YOLO structure
            dst_img = yolo_dir / "images" / split / img_info["file_name"]
            shutil.copy2(src_path, dst_img)

            # Write YOLO label
            label_name = Path(img_info["file_name"]).stem + ".txt"
            dst_label = yolo_dir / "labels" / split / label_name

            anns = image_anns.get(img_id, [])
            with open(dst_label, "w") as f:
                for ann in anns:
                    cx, cy, nw, nh = coco_to_yolo_bbox(
                        ann["bbox"], img_info["width"], img_info["height"]
                    )
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            splits_info[split].append({
                "image_id": img_id,
                "file_name": img_info["file_name"],
                "num_annotations": len(anns),
            })

    # Record test split info (no YOLO conversion needed)
    for img_id in test_ids:
        img_info = id_to_image[img_id]
        anns = image_anns.get(img_id, [])
        splits_info["test"].append({
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "num_annotations": len(anns),
        })

    # ── Save split metadata ───────────────────────────────────────────────────
    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits_info, f, indent=2)

    # ── Print distribution summary ────────────────────────────────────────────
    print(f"\nDataset prepared at {output_dir}")
    print(f"  Train: {len(splits_info['train'])} images (YOLO) → {yolo_dir}")
    print(f"  Val:   {len(splits_info['val'])} images (YOLO) → {yolo_dir}")
    print(f"  Test:  {len(splits_info['test'])} images (COCO) → {test_dir}")
    if skipped:
        print(f"  Skipped (missing): {skipped}")
    print(f"  Split metadata: {splits_path}")

    # Annotation count distribution per split
    for split_name in ("train", "val", "test"):
        counts = [e["num_annotations"] for e in splits_info[split_name]]
        if counts:
            total_anns = sum(counts)
            print(f"  {split_name}: {total_anns} annotations, "
                  f"mean={np.mean(counts):.1f}, "
                  f"range=[{min(counts)}, {max(counts)}]")

    # ── Stamp absolute path into dataset.yaml ──────────────────────────────────
    # Ultralytics resolves relative paths against its own datasets_dir setting,
    # not the YAML file's location — so we must use absolute paths.
    dataset_yaml = TRAINING_ROOT / "configs" / "dataset.yaml"
    if dataset_yaml.exists():
        import yaml

        with open(dataset_yaml) as f:
            ds_cfg = yaml.safe_load(f)

        ds_cfg["path"] = str(yolo_dir.resolve())

        with open(dataset_yaml, "w") as f:
            # Write comment header manually, then dump the rest
            f.write("# Dataset configuration for YOLOv8 fine-tuning\n")
            f.write("# NOTE: 'path' is written as an absolute path by prepare_dataset.py\n")
            f.write("# Ultralytics resolves relative paths against its own datasets_dir setting,\n")
            f.write("# not the YAML file location — so absolute paths are the only reliable option.\n\n")
            yaml.dump(ds_cfg, f, default_flow_style=False, sort_keys=False)

        print(f"  Updated dataset.yaml path → {yolo_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare coin dataset: 3-way split → YOLO (train/val) + COCO (test)"
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
