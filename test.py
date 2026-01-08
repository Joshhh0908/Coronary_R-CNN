# analyze_anchor_lengths.py
import numpy as np
import pandas as pd
from ast import literal_eval

SPACING_MM = 0.2
STRIDE = 16  # FE downsample along D

def is_normal_list(x) -> bool:
    return isinstance(x, (list, tuple)) and any(isinstance(v, str) and v.lower() == "normal" for v in x)

def parse_triplet_lengths_mm(triplet_list):
    """
    triplet_list: ['PID','VESSEL', s_mm, e_mm, val, s_mm, e_mm, val, ...]
    returns list of (len_mm, val)
    """
    out = []
    if not isinstance(triplet_list, (list, tuple)) or len(triplet_list) < 5:
        return out
    if is_normal_list(triplet_list):
        return out

    n = (len(triplet_list) - 2) // 3
    for k in range(n):
        s_mm = triplet_list[2 + 3*k + 0]
        e_mm = triplet_list[2 + 3*k + 1]
        val  = triplet_list[2 + 3*k + 2]
        try:
            s = float(s_mm); e = float(e_mm)
        except Exception:
            continue
        if e > s:
            out.append((e - s, val))
    return out

def main(csv_path: str):
    df = pd.read_csv(csv_path, header=None, index_col=0)
    rows = df.values.tolist()

    lens_slices = []
    lens_feat = []

    for r in rows:
        # your row: pid, cpr, mask, plaque_str, stenosis_str (based on your dataset code)
        plaque_list = literal_eval(r[3])
        sten_list   = literal_eval(r[4])

        # you can use plaque or stenosis lengths; they should match mostly
        for (len_mm, _) in parse_triplet_lengths_mm(plaque_list):
            Ls = len_mm / SPACING_MM
            Lf = Ls / STRIDE
            lens_slices.append(Ls)
            lens_feat.append(Lf)

    lens_slices = np.array(lens_slices, dtype=np.float32)
    lens_feat   = np.array(lens_feat, dtype=np.float32)

    if lens_feat.size == 0:
        print("No lesions found in CSV.")
        return

    print(f"Num lesions: {len(lens_feat)}")
    print("Slice-length stats:")
    for q in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        print(f"  p{q:02d}: {np.percentile(lens_slices, q):.1f} slices")

    print("\nFeature-length stats (divide by 16):")
    for q in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        print(f"  p{q:02d}: {np.percentile(lens_feat, q):.2f} feat")

    # Recommend anchors based on quantiles (rounded to ints, unique, >=1)
    qs = [10, 20, 35, 50, 65, 80, 90]
    anchors = [int(round(np.percentile(lens_feat, q))) for q in qs]
    anchors = sorted(set(max(1, a) for a in anchors))

    # add a couple small anchors to catch tiny lesions (optional)
    for a in [1, 2]:
        if a not in anchors:
            anchors = [a] + anchors

    print("\nRecommended anchor_lengths (feature units):", anchors)
    print("In slice units:", [a * STRIDE for a in anchors])

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
