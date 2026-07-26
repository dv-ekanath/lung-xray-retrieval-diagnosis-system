# =============================================================
# preprocess.py
# Complete image preprocessing for NIH ChestX-ray14 dataset
# Run this ONCE before model.py
# Output: preprocessed_images/ folder + valid_images.csv
# =============================================================

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageStat
from pathlib import Path
import shutil

# ── Config ────────────────────────────────────────────────────
BASE_DIR   = r"C:\Ekanath\College\Sem6\xray-search-system"
OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessed_images")
CSV_IN     = os.path.join(BASE_DIR, "Data_Entry_2017.csv")
CSV_OUT    = os.path.join(BASE_DIR, "valid_images.csv")

# ── Create output folder ──────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# STEP 1 — Read CSV and apply label filters
# =============================================================
print("=" * 60)
print("STEP 1: Reading and filtering Data_Entry_2017.csv")
print("=" * 60)

df = pd.read_csv(CSV_IN)
print(f"Total records in CSV        : {len(df)}")

# ── Filter 1: Remove 'No Finding' ────────────────────────────
# These are healthy lungs — not useful for disease search
before = len(df)
df = df[df["Finding Labels"] != "No Finding"].reset_index(drop=True)
print(f"After removing 'No Finding' : {len(df)}  "
      f"(removed {before - len(df)})")

# ── Filter 2: PA view only ────────────────────────────────────
# PA = Posteroanterior = standard chest X-ray (best quality)
# AP = Anteroposterior = bedside scan (lower quality, different angle)
before = len(df)
df = df[df["View Position"] == "PA"].reset_index(drop=True)
print(f"After keeping PA view only  : {len(df)}  "
      f"(removed {before - len(df)})")

# ── Show what disease labels remain ──────────────────────────
all_labels = set()
for labels in df["Finding Labels"]:
    for l in labels.split("|"):
        all_labels.add(l.strip())
print(f"\nDisease labels remaining    : {sorted(all_labels)}")
print(f"Total images to process     : {len(df)}\n")


# =============================================================
# STEP 2 — Quality check function
# =============================================================
def quality_check(image_path):
    """
    Returns (True, 'ok') if image passes all checks
    Returns (False, reason) if image should be rejected
    """
    try:
        img  = Image.open(image_path).convert("L")
        w, h = img.size

        # Check 1: minimum size
        if w < 256 or h < 256:
            return False, "too_small"

        stat  = ImageStat.Stat(img)
        mean  = stat.mean[0]
        std   = stat.stddev[0]
        arr   = np.array(img)

        # Check 2: too dark (blank/black image)
        if mean < 10:
            return False, "too_dark"

        # Check 3: too bright (overexposed)
        if mean > 245:
            return False, "too_bright"

        # Check 4: no variation (flat/empty image)
        if std < 8:
            return False, "no_variation"

        # Check 5: mostly black pixels (>80%)
        if np.sum(arr < 5) / arr.size > 0.80:
            return False, "mostly_black"

        # Check 6: mostly white pixels (>80%)
        if np.sum(arr > 250) / arr.size > 0.80:
            return False, "mostly_white"

        # Check 7: image not square enough
        # X-rays should be roughly square (ratio not more than 2:1)
        ratio = max(w, h) / min(w, h)
        if ratio > 2.5:
            return False, "bad_aspect_ratio"

        return True, "ok"

    except Exception as e:
        return False, f"open_error"


# =============================================================
# STEP 3 — Enhancement function
# Improves X-ray contrast and sharpness
# =============================================================
def enhance_xray(img):
    """
    Full enhancement pipeline for medical X-rays:
    1. Convert to grayscale (X-rays are single channel)
    2. Auto-contrast (stretches histogram to 0-255)
    3. Sharpen edges (helps ResNet find lung boundaries)
    4. Convert to RGB (ResNet-50 needs 3 channels)
    """
    # Step A: Ensure grayscale
    img = img.convert("L")

    # Step B: Auto-contrast
    # Cuts off darkest 1% and brightest 1% then stretches
    # This fixes both underexposed and overexposed X-rays
    img = ImageOps.autocontrast(img, cutoff=1)

    # Step C: Sharpen
    # 1.0 = original, 2.0 = very sharp
    # 1.5 is ideal — brings out edges without adding noise
    img = ImageEnhance.Sharpness(img).enhance(1.5)

    # Step D: Back to RGB (3 channels for ResNet-50)
    img = img.convert("RGB")

    return img


# =============================================================
# STEP 4 — Find actual file path across subfolders
# =============================================================
def find_image_path(filename):
    for root, dirs, files in os.walk(BASE_DIR):
        if os.path.basename(root) == "images":
            if filename in files:
                return os.path.join(root, filename)
    return None


# =============================================================
# STEP 5 — Main preprocessing loop
# =============================================================
print("=" * 60)
print("STEP 2: Running preprocessing on all filtered images")
print("=" * 60)
print(f"Output folder: {OUTPUT_DIR}\n")

valid_records = []

skip_reasons  = {
    "file_not_found" : 0,
    "too_small"      : 0,
    "too_dark"       : 0,
    "too_bright"     : 0,
    "no_variation"   : 0,
    "mostly_black"   : 0,
    "mostly_white"   : 0,
    "bad_aspect_ratio": 0,
    "open_error"     : 0,
}

total         = len(df)
saved         = 0
skipped       = 0

for i, row in df.iterrows():
    filename = row["Image Index"]

    # ── Find the file ─────────────────────────────────────────
    src_path = find_image_path(filename)

    if src_path is None:
        skip_reasons["file_not_found"] += 1
        skipped += 1
        continue

    # ── Quality check ─────────────────────────────────────────
    valid, reason = quality_check(src_path)

    if not valid:
        if reason in skip_reasons:
            skip_reasons[reason] += 1
        skipped += 1
        continue

    # ── Enhance and save ──────────────────────────────────────
    try:
        img         = Image.open(src_path)
        enhanced    = enhance_xray(img)
        output_path = os.path.join(OUTPUT_DIR, filename)
        enhanced.save(output_path)

        # Record this image as valid
        valid_records.append({
            "image_name"    : filename,
            "category"      : row["Finding Labels"],
            "description"   : "Chest X-ray showing "
                              + row["Finding Labels"].replace("|", " and "),
            "source_url"    : "NIH ChestX-ray14",
            "view_position" : row["View Position"],
            "patient_age"   : row["Patient Age"],
            "patient_gender": row["Patient Gender"],
        })
        saved += 1

    except Exception as e:
        skip_reasons["open_error"] += 1
        skipped += 1
        continue

    # ── Progress every 1000 ───────────────────────────────────
    if (i + 1) % 1000 == 0:
        pct = ((i + 1) / total) * 100
        print(f"  [{i+1:>6}/{total}]  {pct:5.1f}%"
              f"  | Saved: {saved}"
              f"  | Skipped: {skipped}")


# =============================================================
# STEP 6 — Save valid_images.csv
# =============================================================
valid_df = pd.DataFrame(valid_records)
valid_df.to_csv(CSV_OUT, index=False)


# =============================================================
# STEP 7 — Final summary
# =============================================================
print("\n" + "=" * 60)
print("  PREPROCESSING COMPLETE")
print("=" * 60)
print(f"  Total in CSV (after label filter) : {total}")
print(f"  Successfully preprocessed & saved : {saved}")
print(f"  Total skipped                     : {skipped}")
print(f"\n  Skip breakdown:")
for reason, count in skip_reasons.items():
    if count > 0:
        print(f"    {reason:<22}: {count}")
print(f"\n  Output images → {OUTPUT_DIR}")
print(f"  Output CSV    → {CSV_OUT}")
print("=" * 60)
print("\nNext step: run  python model.py")
