import json
import numpy as np
import torch
from check_leakage import compute_dhash_bits, compute_hashes_for_split

with open('dataset_splits_clean.json', 'r') as f:
    splits = json.load(f)

tr_bits = compute_hashes_for_split(splits['train'])
te_bits = compute_hashes_for_split(splits['test'])

t_tr = torch.from_numpy(tr_bits).float()
t_te = torch.from_numpy(te_bits).float()

dist = torch.cdist(t_tr, t_te, p=1).int().numpy()
i_indices, j_indices = np.where(dist == 0)
print(f'Train <-> Test exact dHash matches count: {len(i_indices)}')
for i, j in zip(i_indices, j_indices):
    s1 = splits['train'][i]
    s2 = splits['test'][j]
    print(f"  [{s1['class']}] {s1['filename']} <==> [{s2['class']}] {s2['filename']}")
