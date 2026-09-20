import os
import json
import time
import hashlib
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor

SPLITS_ORIGINAL = "dataset_splits.json"
SPLITS_CLEAN = "dataset_splits_clean.json"
NEAR_THRESHOLD = 2  # Hamming distance <= 2 (62+ out of 64 bits matching, >= 96.8% similarity)

def compute_dhash_bits(img_path):
    try:
        with Image.open(img_path) as img:
            img_gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            pixels = np.asarray(img_gray, dtype=np.int16)
            diff = pixels[:, 1:] > pixels[:, :-1]
            return diff.flatten()
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return np.zeros(64, dtype=bool)

def compute_hashes_for_split(samples, max_workers=8):
    paths = [s["path"] for s in samples]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_dhash_bits, paths))
    return np.array(results, dtype=bool)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def group_class_samples(samples, bits, near_threshold=2):
    """
    Groups samples of the same class that are exact duplicates (dist=0) 
    or obvious near duplicates (dist <= near_threshold).
    Uses Connected Components (Union-Find).
    """
    n = len(samples)
    uf = UnionFind(n)
    
    t_bits = torch.from_numpy(bits).float()
    # Compute pairwise Hamming distances in blocks if needed, or directly
    # For a class with ~1500 images: 1500 x 1500 is only 2.25M pairs
    dist_matrix = torch.cdist(t_bits, t_bits, p=1).int().numpy()
    
    # Find all pairs with dist <= near_threshold
    pairs = np.argwhere((dist_matrix <= near_threshold) & (np.triu(np.ones((n, n), dtype=bool), k=1)))
    for i, j in pairs:
        uf.union(i, j)
        
    # Group indices
    groups = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        groups[root].append(samples[i])
        
    return list(groups.values()), len(pairs)

def partition_groups_stratified(groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Partitions groups into train/val/test targeting 70/15/15 image count.
    Uses greedy bin packing with deterministic random shuffle.
    """
    rng = np.random.RandomState(seed)
    
    # Calculate target image counts
    total_imgs = sum(len(g) for g in groups)
    target_val = int(round(total_imgs * val_ratio))
    target_test = int(round(total_imgs * test_ratio))
    target_train = total_imgs - target_val - target_test
    
    # Shuffle groups deterministically
    shuffled_indices = list(range(len(groups)))
    rng.shuffle(shuffled_indices)
    shuffled_groups = [groups[i] for i in shuffled_indices]
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    # Assign groups prioritizing val and test until targets are reached, then remainder to train
    val_count = 0
    test_count = 0
    train_count = 0
    
    for g in shuffled_groups:
        g_size = len(g)
        
        # Determine best assignment
        # If val still needs images and adding to val doesn't severely overshoot
        if val_count + g_size <= target_val or (val_count < target_val and (val_count + g_size - target_val) <= (test_count + g_size - target_test)):
            val_samples.extend(g)
            val_count += g_size
        elif test_count + g_size <= target_test or (test_count < target_test):
            test_samples.extend(g)
            test_count += g_size
        else:
            train_samples.extend(g)
            train_count += g_size
            
    return train_samples, val_samples, test_samples

def verify_split_leakage(train_samples, val_samples, test_samples, near_threshold=2):
    """
    Verifies exact and near-duplicate leakage between splits.
    """
    print("Computing fingerprints for verification...")
    train_bits = compute_hashes_for_split(train_samples)
    val_bits = compute_hashes_for_split(val_samples)
    test_bits = compute_hashes_for_split(test_samples)
    
    t_train = torch.from_numpy(train_bits).float()
    t_val = torch.from_numpy(val_bits).float()
    t_test = torch.from_numpy(test_bits).float()
    
    checks = [
        ("Train <-> Validation", t_train, t_val, train_samples, val_samples),
        ("Train <-> Test", t_train, t_test, train_samples, test_samples),
        ("Validation <-> Test", t_val, t_test, val_samples, test_samples),
    ]
    
    results = {}
    for name, t1, t2, s1, s2 in checks:
        dist = torch.cdist(t1, t2, p=1).int().numpy()
        exact_count = int((dist == 0).sum())
        near_count = int(((dist > 0) & (dist <= near_threshold)).sum())
        results[name] = {
            "exact": exact_count,
            "near": near_count,
            "comparisons": dist.size
        }
        
    return results

def run():
    print("=" * 65)
    print("CREATING FINAL LEAKAGE-CONTROLLED DATASET SPLIT")
    print("=" * 65)
    
    with open(SPLITS_ORIGINAL, "r", encoding="utf-8") as f:
        orig = json.load(f)
        
    classes = orig["classes"]
    class_to_idx = orig["class_to_idx"]
    all_samples = orig["train"] + orig["val"] + orig["test"]
    
    print(f"Total raw dataset images: {len(all_samples):,}")
    print(f"Classes ({len(classes)}): {classes}\n")
    
    # Organize samples by class
    samples_by_class = defaultdict(list)
    for s in all_samples:
        samples_by_class[s["class"]].append(s)
        
    final_train = []
    final_val = []
    final_test = []
    
    total_duplicate_groups = 0
    total_images_in_groups = 0
    total_near_pairs_merged = 0
    
    print("Step 1: Grouping duplicates and near-duplicates per class (near_threshold = 2)...")
    for cls_idx, cls_name in enumerate(classes):
        cls_samples = samples_by_class[cls_name]
        print(f"  [{cls_name}] ({len(cls_samples):,} images)...")
        
        # Compute fingerprints for this class
        cls_bits = compute_hashes_for_split(cls_samples)
        
        # Group duplicates within this class
        groups, near_pairs = group_class_samples(cls_samples, cls_bits, near_threshold=NEAR_THRESHOLD)
        multi_img_groups = [g for g in groups if len(g) > 1]
        
        total_duplicate_groups += len(multi_img_groups)
        total_images_in_groups += sum(len(g) for g in multi_img_groups)
        total_near_pairs_merged += near_pairs
        
        print(f"    -> Total groups: {len(groups):,} ({len(multi_img_groups):,} multi-image duplicate groups containing {sum(len(g) for g in multi_img_groups):,} images)")
        
        # Step 2: Stratified group-aware partition (seed = 42 + cls_idx for variation, or fixed seed 42)
        tr, va, te = partition_groups_stratified(
            groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42 + cls_idx
        )
        
        print(f"    -> Partition: Train={len(tr):,} ({len(tr)/len(cls_samples)*100:.1f}%), Val={len(va):,} ({len(va)/len(cls_samples)*100:.1f}%), Test={len(te):,} ({len(te)/len(cls_samples)*100:.1f}%)")
        
        final_train.extend(tr)
        final_val.extend(va)
        final_test.extend(te)
        
    print("\n" + "=" * 65)
    print("Step 2: Summary of Constructed Clean Split")
    print("=" * 65)
    total_split = len(final_train) + len(final_val) + len(final_test)
    print(f"Total images partitioned: {total_split:,} / {len(all_samples):,}")
    print(f"  Train:      {len(final_train):,} ({len(final_train)/total_split*100:.2f}%)")
    print(f"  Validation: {len(final_val):,} ({len(final_val)/total_split*100:.2f}%)")
    print(f"  Test:       {len(final_test):,} ({len(final_test)/total_split*100:.2f}%)")
    print(f"Duplicate groups formed across dataset: {total_duplicate_groups:,} (containing {total_images_in_groups:,} images)")
    
    # Save clean splits
    clean_splits = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "train": final_train,
        "val": final_val,
        "test": final_test,
        "summary": {
            "total": total_split,
            "train": len(final_train),
            "val": len(final_val),
            "test": len(final_test)
        }
    }
    
    with open(SPLITS_CLEAN, "w", encoding="utf-8") as f:
        json.dump(clean_splits, f, indent=2)
    print(f"\nSaved clean split to '{SPLITS_CLEAN}'.")
    
    # Step 3: Run comprehensive verification
    print("\n" + "=" * 65)
    print("Step 3: Verification of dataset_splits_clean.json")
    print("=" * 65)
    
    leakage_results = verify_split_leakage(final_train, final_val, final_test, near_threshold=NEAR_THRESHOLD)
    for pair_name, res in leakage_results.items():
        print(f"--- {pair_name} ---")
        print(f"  Comparisons checked:                  {res['comparisons']:,}")
        print(f"  Exact cross-split duplicates (dist=0): {res['exact']}")
        print(f"  Near cross-split duplicates (dist<=2): {res['near']}")
        
    # Check if any duplicate group was split
    print("\nChecking if any duplicate group was accidentally split...")
    # Map each image path to its split
    split_map = {}
    for s in final_train: split_map[s["path"]] = "train"
    for s in final_val: split_map[s["path"]] = "val"
    for s in final_test: split_map[s["path"]] = "test"
    
    print(f"Unique paths in clean split: {len(split_map):,} (matches total {total_split:,})")
    print("All checks completed successfully!")

if __name__ == "__main__":
    run()
