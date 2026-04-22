import sys
import json
import random
from pathlib import Path
from PIL import Image

PLATFORM_TO_DIR = {
    "sd":      "SD_dataset",
    "dall-E3": "DALLE_dataset",
    "GLIDE":   "GLIDE_dataset",
    "im":      "IMAGEN_dataset",
}

# Standalone datasets (not in JSON, just fake/ and real/ folders)
STANDALONE_DATASETS = [
    "compDataset",
    "Gemini_dataset",
]

def process_and_save(src_path, dst_path, size=(256, 256)):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize(size, Image.Resampling.LANCZOS)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def split_standalone(db_root: Path, dataset_name: str, out_dir: Path,
                     train_r=0.70, val_r=0.10, seed=42):
    """Split a standalone fake/real dataset (not JSON-paired) into train/val/test."""
    src = db_root / dataset_name
    counts = {"train": 0, "val": 0, "test": 0}

    for label in ("real", "fake"):
        folder = src / label
        if not folder.is_dir():
            print(f"  [{dataset_name}] No {label}/ folder found, skipping.")
            continue

        paths = [p for p in sorted(folder.iterdir())
                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        rng = random.Random(seed)
        rng.shuffle(paths)

        n = len(paths)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)

        split_map = (
            [("train", p) for p in paths[:n_train]] +
            [("val",   p) for p in paths[n_train:n_train + n_val]] +
            [("test",  p) for p in paths[n_train + n_val:]]
        )

        for split, src_path in split_map:
            prefix = f"{dataset_name}_{src_path.stem}_"
            dst = out_dir / split / label / (prefix + "img.jpg")
            if process_and_save(src_path, dst):
                counts[split] += 1

    print(f"  [{dataset_name}] train={counts['train']} val={counts['val']} test={counts['test']} images added")
    return counts


def main():
    print("Initializing DeepGuardDB_v1 full split: 70% Train, 10% Val, 20% Test")
    print("Sources: DALLE + GLIDE + IMAGEN + SD (JSON-paired) + compDataset + Gemini_dataset (standalone)")

    db_root   = Path("../DeepGuardDB_v1")
    json_path = db_root / "json_files" / "DeepGuardDB.json"
    out_dir   = Path("dataset")
    out_dir.mkdir(exist_ok=True)

    # ── JSON-paired datasets ──────────────────────────────────────────────
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_pairs = []
    skipped = 0
    for item in data:
        platform = item.get("platform", "")
        dataset_dir = PLATFORM_TO_DIR.get(platform)
        if dataset_dir is None:
            skipped += 1
            continue
        real_img = db_root / dataset_dir / "real" / item["real_image_file_name"]
        fake_img = db_root / dataset_dir / "fake" / item["fake_image_file_name"]
        if real_img.exists() and fake_img.exists():
            valid_pairs.append({**item, "_dataset_dir": dataset_dir})
        else:
            skipped += 1

    print(f"Found {len(valid_pairs)} valid JSON pairs ({skipped} skipped).")

    ids = [item["id"] for item in valid_pairs]
    random.seed(42)
    random.shuffle(ids)

    n_total = len(ids)
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.10)

    train_ids = set(ids[:n_train])
    val_ids   = set(ids[n_train:n_train + n_val])

    splits = {"train": 0, "val": 0, "test": 0}
    errors = 0

    for i, item in enumerate(valid_pairs):
        if i % 200 == 0:
            print(f"  JSON processing: {i}/{n_total} ({i/n_total*100:.1f}%)")

        idx = item["id"]
        if idx in train_ids:
            split = "train"
        elif idx in val_ids:
            split = "val"
        else:
            split = "test"

        dataset_dir = item["_dataset_dir"]
        real_src = db_root / dataset_dir / "real" / item["real_image_file_name"]
        fake_src = db_root / dataset_dir / "fake" / item["fake_image_file_name"]

        prefix = f"{dataset_dir}_{item['id']}_"
        real_dst = out_dir / split / "real" / (prefix + "real.jpg")
        fake_dst = out_dir / split / "fake" / (prefix + "fake.jpg")

        ok_r = process_and_save(real_src, real_dst)
        ok_f = process_and_save(fake_src, fake_dst)
        if ok_r and ok_f:
            splits[split] += 1
        else:
            errors += 1

    print(f"\nJSON-paired results:")
    print(f"  Train pairs: {splits['train']} ({splits['train']*2} images)")
    print(f"  Val pairs  : {splits['val']} ({splits['val']*2} images)")
    print(f"  Test pairs : {splits['test']} ({splits['test']*2} images)")

    # ── Standalone datasets ───────────────────────────────────────────────
    print("\nProcessing standalone datasets...")
    for ds_name in STANDALONE_DATASETS:
        ds_path = db_root / ds_name
        if not ds_path.exists():
            print(f"  [{ds_name}] Not found at {ds_path}, skipping.")
            continue
        has_images = any(
            p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for sub in ("fake", "real")
            for p in (ds_path / sub).iterdir()
            if (ds_path / sub).is_dir()
        )
        if not has_images:
            print(f"  [{ds_name}] Folder is empty, skipping (add images first).")
            continue
        split_standalone(db_root, ds_name, out_dir)

    if errors:
        print(f"\n  Errors: {errors} JSON pairs had processing failures")

    print("\nDone! Dataset splits are ready in dataset/train, dataset/val, dataset/test")


if __name__ == "__main__":
    main()
