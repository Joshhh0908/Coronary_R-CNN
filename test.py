import pandas as pd
import ast
from collections import Counter

csv_path = "/home/joshua/Coronary_R-CNN/test_cpr_all26_allbranch_02mm.csv"

# your file format: header=None, index_col=0
df = pd.read_csv(csv_path, header=None, index_col=0)

# stenosis is last column in your example
sten_col = df.columns[-1]

counts = Counter()
bad_rows = []

def parse_list_cell(cell):
    """
    cell is usually a string like:
      "['ARTTS00379', 'LAD', 15.5, 33.5, 3, 38.5, 52.0, 2]"
    Returns a python list or None.
    """
    if not isinstance(cell, str):
        return None
    cell = cell.strip()
    if not cell:
        return None
    try:
        out = ast.literal_eval(cell)
        return out if isinstance(out, list) else None
    except Exception:
        return None

def extract_grades_from_sten_list(sten_list):
    """
    For lesion rows:
      [pid, vessel, s1, e1, g1, s2, e2, g2, ...]
    Grades are at indices 4, 7, 10, ...
    Returns list of int grades (best effort).
    """
    grades = []
    # start at index 4, step 3
    for i in range(4, len(sten_list), 3):
        g = sten_list[i]
        # sometimes might be float-like (e.g. 2.0)
        try:
            gi = int(round(float(g)))
            grades.append(gi)
        except Exception:
            grades.append(None)
    return grades

for idx, row in df.iterrows():
    raw = row[sten_col]
    sten_list = parse_list_cell(raw)

    if sten_list is None:
        counts["parse_fail"] += 1
        bad_rows.append((idx, raw, "parse_fail"))
        continue

    # normal example: ['APNHC00726','LCX','normal']
    if len(sten_list) == 3 and isinstance(sten_list[-1], str) and sten_list[-1].lower() == "normal":
        counts["normal"] += 1
        continue

    # otherwise we expect lesion format: at least [pid, vessel, s, e, grade]
    if len(sten_list) < 5:
        counts["too_short"] += 1
        bad_rows.append((idx, sten_list, "too_short"))
        continue

    grades = extract_grades_from_sten_list(sten_list)

    if not grades:
        counts["no_grades_found"] += 1
        bad_rows.append((idx, sten_list, "no_grades_found"))
        continue

    # check each grade
    ok = True
    for g in grades:
        if g is None:
            counts["grade_not_number"] += 1
            bad_rows.append((idx, sten_list, "grade_not_number"))
            ok = False
            break
        if g == 0:
            counts["grade_is_zero"] += 1
            bad_rows.append((idx, sten_list, "grade_is_zero"))
            ok = False
            break
        if g < 1 or g > 5:
            counts["grade_out_of_range"] += 1
            bad_rows.append((idx, sten_list, f"grade_out_of_range:{g}"))
            ok = False
            break

    if ok:
        counts["lesion_ok"] += 1

print("==== SUMMARY ====")
total = len(df)
print("Total rows:", total)
for k, v in counts.most_common():
    print(f"{k:20s}: {v}")

print("\n==== BAD EXAMPLES (up to 20) ====")
for ex in bad_rows[:20]:
    idx, val, reason = ex
    print(f"\nindex={idx} reason={reason}\nvalue={val}")
