import os
import time
import json
import numpy as np
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor

SPLITS_FILE = "dataset_splits.json"

def compute_dhash_bits(img_path):
    """
    Computes a 64-bit difference hash (dHash) as a boolean numpy array of shape (64,).
    Extremely fast, robust perceptual image fingerprint.
    """
    try:
        with Image.open(img_path) as img:
            # Convert to grayscale and resize to 9x8
            img_gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            pixels = np.asarray(img_gray, dtype=np.int16)
            # Difference between adjacent horizontal pixels
            diff = pixels[:, 1:] > pixels[:, :-1]  # Shape: (8, 8)
            return diff.flatten()  # 64 booleans
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return np.zeros(64, dtype=bool)

def compute_phash_bits(img_path):
    """
    Computes a 64-bit perceptual hash (pHash) using discrete cosine transform.
    """
    from scipy.fftpack import dct
    try:
        with Image.open(img_path) as img:
            img_gray = img.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
            pixels = np.asarray(img_gray, dtype=np.float32)
            dct_rows = dct(pixels, axis=0, norm='ortho')
            dct_2d = dct(dct_rows, axis=1, norm='ortho')
            # Top-left 8x8 low frequencies, excluding (0,0) DC term
            dct_low = dct_2d[:8, :8]
            median_val = np.median(dct_low.flatten()[1:])
            diff = dct_low > median_val
            return diff.flatten()
    except Exception as e:
        return np.zeros(64, dtype=bool)

def compute_hashes_for_split(samples, hash_fn, max_workers=8):
    paths = [s["path"] for s in samples]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(hash_fn, paths))
    return np.array(results, dtype=bool)

def analyze_pair(split1_name, split1_samples, split1_bits, 
                 split2_name, split2_samples, split2_bits, 
                 near_threshold=2):
    """
    Compares two splits using Hamming distance.
    Hamming distance = 0 -> exact perceptual duplicate
    Hamming distance in [1, near_threshold] -> obvious near-duplicate
    """
    t1 = torch.from_numpy(split1_bits).float()
    t2 = torch.from_numpy(split2_bits).float()
    
    # Pairwise L1 distance on boolean vectors equals Hamming distance
    # shape: (N1, N2)
    hamming = torch.cdist(t1, t2, p=1).int().numpy()
    
    exact_matches = []
    near_matches = []
    
    n1, n2 = hamming.shape
    for i in range(n1):
        # Exact duplicates: dist == 0
        exact_indices = np.where(hamming[i] == 0)[0]
        for j in exact_indices:
            exact_matches.append({
                "item1": split1_samples[i],
                "item2": split2_samples[j],
                "dist": 0
            })
            
        # Near duplicates: 0 < dist <= near_threshold
        near_indices = np.where((hamming[i] > 0) & (hamming[i] <= near_threshold))[0]
        for j in near_indices:
            near_matches.append({
                "item1": split1_samples[i],
                "item2": split2_samples[j],
                "dist": int(hamming[i, j])
            })
            
    return exact_matches, near_matches

def run_sanity_check():
    print("=" * 65)
    print("DATA LEAKAGE & DUPLICATE SANITY CHECK ON DATASET SPLITS")
    print("=" * 65)
    
    if not os.path.exists(SPLITS_FILE):
        print(f"Error: {SPLITS_FILE} not found.")
        return
        
    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    
    print(f"Loaded splits from '{SPLITS_FILE}':")
    print(f"  Train:      {len(train_samples):,} images")
    print(f"  Validation: {len(val_samples):,} images")
    print(f"  Test:       {len(test_samples):,} images")
    print(f"  Total:      {len(train_samples) + len(val_samples) + len(test_samples):,} images\n")
    
    t0 = time.time()
    print("Computing 64-bit dHash perceptual fingerprints for all splits...")
    train_dhash = compute_hashes_for_split(train_samples, compute_dhash_bits)
    val_dhash = compute_hashes_for_split(val_samples, compute_dhash_bits)
    test_dhash = compute_hashes_for_split(test_samples, compute_dhash_bits)
    print(f"Fingerprinting completed in {time.time() - t0:.2f}s.\n")
    
    pairs_to_check = [
        ("Train", train_samples, train_dhash, "Validation", val_samples, val_dhash),
        ("Train", train_samples, train_dhash, "Test", test_samples, test_dhash),
        ("Validation", val_samples, val_dhash, "Test", test_samples, test_dhash),
    ]
    
    # We will test with near_threshold = 2 (Hamming distance <= 2 out of 64 bits = >= 96.8% visual identity)
    # Also verify with near_threshold = 4 (>= 93.7% identity)
    near_threshold = 2
    
    all_exact = {}
    all_near = {}
    
    for s1_name, s1_samples, s1_bits, s2_name, s2_samples, s2_bits in pairs_to_check:
        pair_key = f"{s1_name} <-> {s2_name}"
        exact, near = analyze_pair(
            s1_name, s1_samples, s1_bits, 
            s2_name, s2_samples, s2_bits, 
            near_threshold=near_threshold
        )
        all_exact[pair_key] = exact
        all_near[pair_key] = near
        
        print(f"--- Checking: {pair_key} ---")
        print(f"  Comparisons checked: {len(s1_samples):,} x {len(s2_samples):,} = {len(s1_samples)*len(s2_samples):,} pairs")
        print(f"  Exact Duplicates (dist=0):        {len(exact)}")
        print(f"  Near-Duplicates (1 <= dist <= {near_threshold}): {len(near)}")
        
        if len(exact) > 0:
            print("  Sample Exact Matches:")
            for m in exact[:5]:
                print(f"    - [{m['item1']['class']}] {m['item1']['filename']}  <==>  [{m['item2']['class']}] {m['item2']['filename']}")
                
        if len(near) > 0:
            print(f"  Sample Near Matches (Hamming dist <= {near_threshold}):")
            for m in near[:5]:
                print(f"    - dist={m['dist']}: [{m['item1']['class']}] {m['item1']['filename']}  <==>  [{m['item2']['class']}] {m['item2']['filename']}")
        print()
        
    # Intra-split duplicates check (for context)
    print("--- Intra-Split Duplicate Check (within same split) ---")
    for s_name, s_samples, s_bits in [("Train", train_samples, train_dhash), 
                                      ("Validation", val_samples, val_dhash), 
                                      ("Test", test_samples, test_dhash)]:
        t = torch.from_numpy(s_bits).float()
        hamming = torch.cdist(t, t, p=1).int().numpy()
        # Zero out diagonal
        np.fill_diagonal(hamming, 999)
        exact_intra = np.sum(hamming == 0) // 2
        near_intra = np.sum((hamming > 0) & (hamming <= near_threshold)) // 2
        print(f"  {s_name}: {exact_intra} exact duplicate pairs, {near_intra} near-duplicate pairs")
    print()
    
    total_exact_across = sum(len(v) for v in all_exact.values())
    total_near_across = sum(len(v) for v in all_near.values())
    
    print("=" * 65)
    print("SUMMARY FINDINGS")
    print("=" * 65)
    print(f"Total Exact Duplicates across splits:  {total_exact_across}")
    print(f"Total Near-Duplicates across splits:   {total_near_across} (dist <= {near_threshold})")
    
    for pair_key in all_exact:
        print(f"  * {pair_key:22}: Exact = {len(all_exact[pair_key])}, Near = {len(all_near[pair_key])}")
        
    is_clean = (total_exact_across == 0) and (total_near_across == 0)
    print(f"\nOverall Split Cleanliness Assessment: {'CLEAN (No Leakage Detected)' if is_clean else 'LEAKAGE / OVERLAP DETECTED'}")
    print("=" * 65)

if __name__ == "__main__":
    run_sanity_check()
