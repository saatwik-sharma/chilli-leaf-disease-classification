import os
import hashlib
from collections import defaultdict, Counter
from PIL import Image

DATASET_ROOT = "Chilli Leaf Disease Image Dataset for Classificati"

def inspect():
    print(f"Inspecting root directory: {DATASET_ROOT}")
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: {DATASET_ROOT} does not exist yet.")
        return

    # 1. Structure
    entries = sorted(os.listdir(DATASET_ROOT))
    print("\n--- 1. Root Directory Contents ---")
    for e in entries:
        full = os.path.join(DATASET_ROOT, e)
        is_d = os.path.isdir(full)
        size_mb = os.path.getsize(full) / (1024 * 1024) if not is_d else 0
        print(f"  {'[DIR] ' if is_d else '[FILE]'} {e} {'(' + f'{size_mb:.2f} MB' + ')' if not is_d else ''}")

    subdirs = [e for e in entries if os.path.isdir(os.path.join(DATASET_ROOT, e))]
    
    # 2, 3, 4. Classes & Images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    total_images = 0
    class_counts = {}
    corrupted_files = []
    
    formats_count = Counter()
    modes_count = Counter()
    dimensions_count = Counter()
    dimensions_by_class = defaultdict(set)
    
    SKIP_DUPLICATE_HASH = True
    
    # Duplicate tracking: hash -> list of (class_name, file_path)
    file_hashes = defaultdict(list)
    
    all_files_by_ext = Counter()

    for cls in subdirs:
        cls_dir = os.path.join(DATASET_ROOT, cls)
        cls_files = sorted(os.listdir(cls_dir))
        cls_img_count = 0
        print(f"Checking class: {cls} ({len(cls_files)} items)...")
        
        for fname in cls_files:
            fpath = os.path.join(cls_dir, fname)
            if not os.path.isfile(fpath):
                continue
                
            ext = os.path.splitext(fname)[1].lower()
            all_files_by_ext[ext] += 1
            
            if ext in image_extensions:
                cls_img_count += 1
                total_images += 1
                
                # Check readability & corruption
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    with Image.open(fpath) as img:
                        fmt = img.format
                        mode = img.mode
                        size = img.size
                        img.load()
                        formats_count[fmt or img.format] += 1
                        modes_count[mode] += 1
                        dimensions_count[size] += 1
                        dimensions_by_class[cls].add(size)
                except Exception as ex:
                    corrupted_files.append((fpath, str(ex)))

                # Hash for duplicates (skipped for performance)
                if not SKIP_DUPLICATE_HASH:
                    try:
                        h = hashlib.sha256()
                        with open(fpath, "rb") as f:
                            while chunk := f.read(1024 * 1024):
                                h.update(chunk)
                        digest = h.hexdigest()
                        file_hashes[digest].append((cls, fname, fpath))
                    except Exception as ex:
                        print(f"Error hashing {fpath}: {ex}")
                    
        class_counts[cls] = cls_img_count

    print("\n--- Summary of Findings ---")
    print(f"Total images found: {total_images}")
    print(f"Identified classes ({len(class_counts)}):")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count} images")

    print("\nFile Extensions:")
    for ext, count in all_files_by_ext.items():
        print(f"  {ext}: {count}")

    print("\nImage Formats (Pillow detected):")
    for fmt, count in formats_count.items():
        print(f"  {fmt}: {count}")

    print("\nColor Modes:")
    for mode, count in modes_count.items():
        print(f"  {mode}: {count}")

    print(f"\nUnique Dimensions count: {len(dimensions_count)}")
    print("Most common dimensions (width, height):")
    for dim, count in dimensions_count.most_common(10):
        print(f"  {dim[0]}x{dim[1]}: {count} images ({count/total_images*100:.1f}%)")
    if len(dimensions_count) > 10:
        print(f"  ... and {len(dimensions_count) - 10} other unique dimensions")

    widths = [dim[0] for dim in dimensions_count.elements()]
    heights = [dim[1] for dim in dimensions_count.elements()]
    if widths and heights:
        print(f"Dimension ranges: Width [{min(widths)} - {max(widths)}], Height [{min(heights)} - {max(heights)}]")

    print(f"\nCorrupted / Unreadable Images: {len(corrupted_files)}")
    for cf, err in corrupted_files:
        print(f"  CORRUPTED: {cf} -> {err}")

    # Duplicates
    if SKIP_DUPLICATE_HASH:
        print("\nSHA-256 Duplicate Check: Skipped")
    else:
        exact_duplicates = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
        print(f"\nUnique SHA-256 Hashes: {len(file_hashes)}")
        print(f"Sets of Duplicate Images: {len(exact_duplicates)}")
        
        intra_class_dups = 0
        inter_class_dups = 0
        dup_details = []

        for h, paths in exact_duplicates.items():
            classes_in_dup = set(p[0] for p in paths)
            if len(classes_in_dup) > 1:
                inter_class_dups += 1
                dup_details.append(("CROSS-CLASS", paths))
            else:
                intra_class_dups += 1
                dup_details.append(("INTRA-CLASS", paths))

        print(f"  - Intra-class duplicates (same class, duplicate file): {intra_class_dups} groups")
        print(f"  - Cross-class duplicates (different classes, identical file): {inter_class_dups} groups")
        
        if exact_duplicates:
            print("\nSample duplicate groups (first 10):")
            for dup_type, paths in dup_details[:10]:
                print(f"  [{dup_type}] ({len(paths)} copies):")
                for cls, fname, _ in paths:
                    print(f"     - Class '{cls}': {fname}")

if __name__ == "__main__":
    inspect()
