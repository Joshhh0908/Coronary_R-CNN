# Look at the class distribution in your GT
from collections import Counter

from cached_ds import CachedWindowDataset

ds = CachedWindowDataset("new_data/cpr_cache_pt", "test")
plaq_classes = []
for i in range(len(ds)):
    _, targets = ds[i]
    if len(targets["boxes"]) > 0:
        plaq_classes.extend(targets["plaque"].tolist())

print("Plaque class distribution:", Counter(plaq_classes))

import pandas as pd
from ast import literal_eval
from dataprep import parse_triplet_intervals

test = pd.read_csv("/home/joshua/Coronary_R-CNN/new_data/test_cpr_all26_allbranch_02to04mm_review4.csv")
# train = pd.read_csv("/home/joshua/Coronary_R-CNN/new_data/train_val_cpr_all26_allbranch_02to04mm_review4.csv")
# print(test.iloc[:, 5])
test_plaq = test.iloc[:, 4].tolist()
# train_plaq = train.iloc[:, 5].tolist()
# train_plaq = train.iloc[:, 5].tolist()

val_counter = {1:0, 2:0, 3:0}

for plaq in test_plaq:
    intervals = parse_triplet_intervals(literal_eval(plaq), 99999999)
    for interval in intervals:
        val = interval[2]
        val_counter[val] = val_counter.get(val, 0) + 1
print("Test plaque class distribution:", val_counter)