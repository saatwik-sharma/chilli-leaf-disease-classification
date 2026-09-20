import json
import os
import hashlib
from PIL import Image

SPLIT_FILE = "dataset_splits_clean.json"

EXPECTED = {
    "train": 6173,
    "val": 1322,
    "test": 1322
}

CLASSES = [
    "Bacterial_Spot",
    "Cercospora_Leaf_Spot",
    "Curl_Virus",
    "Healthy_Leaf",
    "Nutrition_Deficiency",
    "Powdery_Mildew"
]

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    splits = json.load(f)

print("=" * 60)
print("CLEAN DATASET SPLIT VERIFICATION")
print("=" * 60)

# --------------------------------------------------
# 1. Count verification
# --------------------------------------------------

print("\n[1] Split Counts")

total = 0

for split in ["train", "val", "test"]:
    count = len(splits[split])
    total += count
    print(f"{split:>5}: {count}")

print(f"total: {total}")

count_ok = all(len(splits[s]) == EXPECTED[s] for s in EXPECTED) and total == 8817

print("Count check:", "PASS" if count_ok else "FAIL")

# --------------------------------------------------
# 2. Check every image occurs exactly once
# --------------------------------------------------

print("\n[2] Image Assignment Check")

all_paths = []

for split in ["train", "val", "test"]:
    for item in splits[split]:
        path = os.path.normcase(os.path.abspath(item["path"]))
        all_paths.append(path)

unique_paths = set(all_paths)

print("Total image entries :", len(all_paths))
print("Unique image paths  :", len(unique_paths))

assignment_ok = len(all_paths) == len(unique_paths)

print(
    "Assignment check:",
    "PASS" if assignment_ok else "FAIL"
)

# --------------------------------------------------
# 3. Class distribution
# --------------------------------------------------

print("\n[3] Class Distribution")

for split in ["train", "val", "test"]:
    counts = {c: 0 for c in CLASSES}

    for item in splits[split]:
        label = item.get("class")

        if label in counts:
            counts[label] += 1

    print(f"\n{split.upper()}")

    for c in CLASSES:
        print(f"  {c:<25} {counts[c]}")

# --------------------------------------------------
# 4. Cross-split exact file duplicate check
# --------------------------------------------------

print("\n[4] Cross-Split Exact File Duplicate Check")

seen = {}
cross_duplicates = []

for split in ["train", "val", "test"]:

    for item in splits[split]:

        path = os.path.normcase(os.path.abspath(item["path"]))

        if path in seen:
            if seen[path] != split:
                cross_duplicates.append(
                    (path, seen[path], split)
                )
        else:
            seen[path] = split

print("Cross-split duplicate paths:", len(cross_duplicates))

if cross_duplicates:
    for x in cross_duplicates[:10]:
        print(x)

# --------------------------------------------------
# 5. Lightweight dHash
# --------------------------------------------------

print("\n[5] Cross-Split Identical dHash Check")

def dhash(path):
    try:
        img = Image.open(path).convert("L")
        img = img.resize((9, 8))

        pixels = list(img.getdata())

        bits = []

        for row in range(8):
            for col in range(8):
                left = pixels[row * 9 + col]
                right = pixels[row * 9 + col + 1]

                bits.append("1" if left > right else "0")

        return hex(int("".join(bits), 2))[2:].zfill(16)

    except Exception:
        return None


hashes = {}
cross_hash_duplicates = []

for split in ["train", "val", "test"]:

    print(f"Processing {split}...")

    for item in splits[split]:

        path = item["path"]

        h = dhash(path)

        if h is None:
            continue

        if h in hashes:

            previous_split, previous_path = hashes[h]

            if previous_split != split:
                cross_hash_duplicates.append(
                    (h, previous_split, previous_path, split, path)
                )

        else:
            hashes[h] = (split, path)

print(
    "Cross-split identical dHash groups:",
    len(cross_hash_duplicates)
)

if cross_hash_duplicates:
    print("\nExamples:")

    for x in cross_hash_duplicates[:10]:
        print(x)

# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

print("\n" + "=" * 60)

if (
    count_ok
    and assignment_ok
    and len(cross_duplicates) == 0
    and len(cross_hash_duplicates) == 0
):
    print("CLEAN SPLIT VERIFIED")
else:
    print("VERIFICATION FAILED")

print("=" * 60)