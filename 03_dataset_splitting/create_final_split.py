"""
create_final_split.py  (v4 - fast move-based optimisation)
===========================================================
Produces dataset_splits_final.json with:
  Train ~70 %  Val ~15 %  Test ~15 %

All dHash-connected near-duplicate groups (Hamming <= 2) are kept
within a single split.  No original images are touched.
SEED = 42  ->  fully reproducible.
"""

import json
import os
import random
import sys
from collections import defaultdict
from PIL import Image

SEED   = 42
INPUT  = "dataset_splits_clean.json"
OUTPUT = "dataset_splits_final.json"
SPLITS = ["train", "val", "test"]
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}

random.seed(SEED)

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ──────────────────────────────────────────────────────────────
# 1. Load
# ──────────────────────────────────────────────────────────────
with open(INPUT, "r", encoding="utf-8") as f:
    raw = json.load(f)

all_items = []
for split in SPLITS:
    all_items.extend(raw[split])

N = len(all_items)
log(f"Total images loaded: {N}")

classes      = raw["classes"]
class_to_idx = raw["class_to_idx"]

# ──────────────────────────────────────────────────────────────
# 2. dHash (64-bit)
# ──────────────────────────────────────────────────────────────
def dhash(path):
    img = Image.open(path).convert("L").resize((9, 8))
    px  = list(img.getdata())
    val = 0
    for r in range(8):
        for c in range(8):
            val <<= 1
            if px[r * 9 + c] > px[r * 9 + c + 1]:
                val |= 1
    return val

def hamming(a, b):
    return bin(a ^ b).count("1")

# ──────────────────────────────────────────────────────────────
# 3. BK-tree (iterative)
# ──────────────────────────────────────────────────────────────
class BKTree:
    def __init__(self):
        self.root = None
    def insert(self, value, idx):
        if self.root is None:
            self.root = {"v": value, "ids": [idx], "ch": {}}
            return
        node = self.root
        while True:
            d = hamming(value, node["v"])
            if d == 0:
                node["ids"].append(idx)
                return
            if d not in node["ch"]:
                node["ch"][d] = {"v": value, "ids": [idx], "ch": {}}
                return
            node = node["ch"][d]
    def query(self, value, radius=2):
        if self.root is None:
            return []
        results, stack = [], [self.root]
        while stack:
            node = stack.pop()
            d = hamming(value, node["v"])
            if d <= radius:
                results.extend(node["ids"])
            lo, hi = max(0, d - radius), d + radius
            for dist, child in node["ch"].items():
                if lo <= dist <= hi:
                    stack.append(child)
        return results

# ──────────────────────────────────────────────────────────────
# 4. Compute hashes
# ──────────────────────────────────────────────────────────────
log("\nCalculating dHashes ...")
hashes = []
for i, item in enumerate(all_items):
    if i % 500 == 0:
        log(f"  {i}/{N}")
    hashes.append(dhash(item["path"]))
log("dHash calculation complete.")

# ──────────────────────────────────────────────────────────────
# 5. Union-Find connected components
# ──────────────────────────────────────────────────────────────
parent = list(range(N))

def find(x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

log("\nGrouping near-duplicates (Hamming <= 2) ...")
bk = BKTree()
for i, h in enumerate(hashes):
    for j in bk.query(h, 2):
        union(i, j)
    bk.insert(h, i)

raw_groups = defaultdict(list)
for i in range(N):
    raw_groups[find(i)].append(i)

groups = list(raw_groups.values())
multi_groups = [g for g in groups if len(g) > 1]
largest      = max(len(g) for g in groups)

log(f"Total duplicate-aware groups : {len(groups)}")
log(f"Groups with multiple images  : {len(multi_groups)}")
log(f"Largest group                : {largest}")

# ──────────────────────────────────────────────────────────────
# 6. Annotate groups
# ──────────────────────────────────────────────────────────────
total_per_class = defaultdict(int)
for item in all_items:
    total_per_class[item["class"]] += 1

group_info = []
for g in groups:
    cc = defaultdict(int)
    for idx in g:
        cc[all_items[idx]["class"]] += 1
    group_info.append({"indices": g, "cc": dict(cc), "size": len(g)})

random.shuffle(group_info)
group_info.sort(key=lambda x: x["size"], reverse=True)

# ──────────────────────────────────────────────────────────────
# 7. Phase 1 – Greedy (normalised remaining-capacity first)
#
#  Score for assigning group g to split s:
#    = (target_n[s] - current_n[s] - g.size) / target_n[s]
#      + W_CLASS * sum_c( (t_sc - cc_sc - g_c) / t_sc )
#
#  Higher = better.  Train has the biggest quota, so it wins
#  the early assignments and naturally fills to ~70 %.
# ──────────────────────────────────────────────────────────────
target_n = {s: round(N * TARGET[s]) for s in SPLITS}
target_n["test"] = N - target_n["train"] - target_n["val"]

log(f"\nTarget counts -> train: {target_n['train']}  "
    f"val: {target_n['val']}  test: {target_n['test']}")

current_n  = {s: 0 for s in SPLITS}
current_cc = {s: defaultdict(int) for s in SPLITS}
# Each group is stored with its assigned split index
grp_split  = [None] * len(group_info)   # grp_split[i] -> split name

W_CLASS = 0.05   # tiny class nudge; size must dominate

for gi, g in enumerate(group_info):
    best_split = None
    best_score = float("-inf")
    for s in SPLITS:
        remaining_frac = (target_n[s] - current_n[s] - g["size"]) / target_n[s]
        class_bonus = 0.0
        for cls, cnt in g["cc"].items():
            t = total_per_class[cls] * TARGET[s]
            class_bonus += (t - current_cc[s][cls] - cnt) / max(t, 1)
        score = remaining_frac + W_CLASS * class_bonus
        if score > best_score:
            best_score = score
            best_split = s
    grp_split[gi] = best_split
    current_n[best_split] += g["size"]
    for cls, cnt in g["cc"].items():
        current_cc[best_split][cls] += cnt

log(f"After greedy  -> train: {current_n['train']}  "
    f"val: {current_n['val']}  test: {current_n['test']}")

# ──────────────────────────────────────────────────────────────
# 8. Phase 2 – Move-based local optimisation  (O(n) per pass)
#
#  Instead of trying all O(n^2) pairwise swaps (too slow with
#  ~6400 groups), we do single-group MOVES: for each group, try
#  reassigning it to each of the other two splits and accept the
#  move if it reduces the joint objective J.
#
#  J = W_SIZE * sum_s |n_s - target_n_s| / N
#    + W_CLASS * sum_s sum_c |cc_sc - t_sc| / t_sc
# ──────────────────────────────────────────────────────────────
W_SIZE = 5.0

def objective():
    j = 0.0
    for s in SPLITS:
        j += W_SIZE * abs(current_n[s] - target_n[s]) / N
        for cls in classes:
            t  = total_per_class[cls] * TARGET[s]
            j += W_CLASS * abs(current_cc[s][cls] - t) / max(t, 1)
    return j

log("\nRunning move optimisation ...")
MAX_ITERS = 50
itr       = 0

while itr < MAX_ITERS:
    itr       += 1
    base_j     = objective()
    moves_made = 0

    for gi, g in enumerate(group_info):
        src = grp_split[gi]
        best_dst   = None
        best_new_j = base_j

        for dst in SPLITS:
            if dst == src:
                continue

            # Apply move
            current_n[src] -= g["size"]
            current_n[dst] += g["size"]
            for cls, cnt in g["cc"].items():
                current_cc[src][cls] -= cnt
                current_cc[dst][cls] += cnt

            new_j = objective()

            if new_j < best_new_j:
                best_new_j = new_j
                best_dst   = dst

            # Revert
            current_n[src] += g["size"]
            current_n[dst] -= g["size"]
            for cls, cnt in g["cc"].items():
                current_cc[src][cls] += cnt
                current_cc[dst][cls] -= cnt

        if best_dst is not None:
            # Accept best move
            current_n[src] -= g["size"]
            current_n[best_dst] += g["size"]
            for cls, cnt in g["cc"].items():
                current_cc[src][cls] -= cnt
                current_cc[best_dst][cls] += cnt
            grp_split[gi] = best_dst
            base_j        = best_new_j
            moves_made   += 1

    log(f"  Iter {itr}: moves={moves_made}  "
        f"train={current_n['train']}  val={current_n['val']}  test={current_n['test']}  J={base_j:.6f}")

    if moves_made == 0:
        log(f"  No moves improved objective. Stopping at iteration {itr}.")
        break

log(f"Move optimisation done ({itr} iteration(s)).")
log(f"Final counts  -> train: {current_n['train']}  "
    f"val: {current_n['val']}  test: {current_n['test']}")

# ──────────────────────────────────────────────────────────────
# 9. Build final lists
# ──────────────────────────────────────────────────────────────
final = {s: [] for s in SPLITS}
for gi, g in enumerate(group_info):
    s = grp_split[gi]
    for idx in g["indices"]:
        final[s].append(all_items[idx])

for s in SPLITS:
    random.shuffle(final[s])

# ──────────────────────────────────────────────────────────────
# 10. Save
# ──────────────────────────────────────────────────────────────
out_data = {
    "classes":      classes,
    "class_to_idx": class_to_idx,
    "train":        final["train"],
    "val":          final["val"],
    "test":         final["test"],
}
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)

# ──────────────────────────────────────────────────────────────
# 11. Report
# ──────────────────────────────────────────────────────────────
log("\n" + "=" * 65)
log("FINAL DUPLICATE-AWARE SPLIT")
log("=" * 65)
total_saved = sum(len(final[s]) for s in SPLITS)
for s in SPLITS:
    cnt = len(final[s])
    pct = cnt / total_saved * 100
    log(f"\n{s.upper():5} : {cnt}  ({pct:.1f} %)")
    cc = defaultdict(int)
    for item in final[s]:
        cc[item["class"]] += 1
    for cls in classes:
        log(f"    {cls:<30} {cc[cls]}")

log(f"\nTotal duplicate-aware groups : {len(groups)}")
log(f"Groups with multiple images  : {len(multi_groups)}")
log(f"Largest group                : {largest}")
log(f"\nTotal saved  : {total_saved}")
log(f"Train        : {len(final['train'])}  ({len(final['train'])/total_saved*100:.1f} %)")
log(f"Val          : {len(final['val'])}  ({len(final['val'])/total_saved*100:.1f} %)")
log(f"Test         : {len(final['test'])}  ({len(final['test'])/total_saved*100:.1f} %)")
log(f"Output file  : {os.path.abspath(OUTPUT)}")

# ──────────────────────────────────────────────────────────────
# 12. Automatic verification
# ──────────────────────────────────────────────────────────────
log("\n" + "=" * 65)
log("AUTOMATIC VERIFICATION")
log("=" * 65)
errors = []

# A. Total count
if total_saved != 8817:
    errors.append(f"A FAIL: total = {total_saved}, expected 8817")
else:
    log(f"A. Total = {total_saved} - OK")

# B. Every image exactly once
all_paths  = [item["path"] for s in SPLITS for item in final[s]]
path_count = defaultdict(int)
for p in all_paths:
    path_count[p] += 1
dupes   = {p: c for p, c in path_count.items() if c > 1}
missing = set(item["path"] for item in all_items) - set(all_paths)
if dupes:
    errors.append(f"B FAIL: {len(dupes)} paths appear more than once")
elif missing:
    errors.append(f"B FAIL: {len(missing)} images missing")
else:
    log("B. Every image appears exactly once - OK")

# C. No path duplicated across splits
seen  = {}
cross = set()
for s in SPLITS:
    for item in final[s]:
        p = item["path"]
        if p in seen and seen[p] != s:
            cross.add(p)
        seen[p] = s
if cross:
    errors.append(f"C FAIL: {len(cross)} paths appear in multiple splits")
else:
    log("C. No duplicate path across splits - OK")

# D. No identical dHash across splits
log("D. Checking dHash uniqueness across splits ...")
hash_split  = {}
dhash_cross = []
for s in SPLITS:
    for item in final[s]:
        h = dhash(item["path"])
        if h in hash_split and hash_split[h] != s:
            dhash_cross.append((item["path"], s, hash_split[h]))
        else:
            hash_split[h] = s
if dhash_cross:
    errors.append(f"D FAIL: {len(dhash_cross)} identical dHash values across splits")
else:
    log("D. No identical dHash across splits - OK")

# E. All Hamming<=2 groups contained within one split
log("E. Verifying Hamming <= 2 group integrity ...")
path_to_split = {item["path"]: s for s in SPLITS for item in final[s]}
idx_to_split  = {i: path_to_split[all_items[i]["path"]] for i in range(N)}
violations    = sum(
    1 for g in groups
    if len(set(idx_to_split[i] for i in g)) > 1
)
if violations:
    errors.append(f"E FAIL: {violations} group(s) span multiple splits")
else:
    log("E. All Hamming <= 2 groups contained within one split - OK")

log("")
if not errors:
    log("FINAL SPLIT VERIFIED")
else:
    log("FINAL SPLIT FAILED")
    for e in errors:
        log(f"  * {e}")
    sys.exit(1)
