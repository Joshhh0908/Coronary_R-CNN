import pandas as pd
import SimpleITK as sitk
import numpy as np

csv_path = "/home/joshua/Coronary_R-CNN/test_cpr_all26_allbranch_02mm.csv"

df = pd.read_csv(csv_path, header=None, index_col=0)
rows = df.values.tolist()

lengths = []
for r in rows:
    cpr_path = r[1]  # patient_id is r[0], cpr is r[1] in your format
    img = sitk.ReadImage(cpr_path)
    size_xyz = img.GetSize()          # (X, Y, Z)
    lengths.append(size_xyz[2])       # Z is depth

lengths = np.array(lengths)
print("count:", len(lengths))
print("min/median/max:", lengths.min(), int(np.median(lengths)), lengths.max())
for p in [50, 75, 80, 90, 95]:
    print(f"P{p}:", int(np.percentile(lengths, p)))
