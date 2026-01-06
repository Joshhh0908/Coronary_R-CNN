import pandas as pd
import SimpleITK as sitk

csv_path = "/home/joshua/Coronary_R-CNN/test_cpr_all26_allbranch_02mm.csv"
df = pd.read_csv(csv_path, header=None, index_col=0)
rows = df.values.tolist()

# print(df.count())



# count = 0

# for r in rows:
#     cpr_path = r[2]
#     itk = sitk.ReadImage(cpr_path)
#     vol = sitk.GetArrayFromImage(itk)  # [D,H,W]
#     if vol.shape[0] > 768:
#         count += 1

# print(count)


for r in rows[:1]:
    print(r)
    sten = (r[-1])
    print(type(sten))
    print(sten)