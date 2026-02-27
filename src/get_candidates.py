#!/usr/bin/env python3
import os
import argparse
import csv
import glob

parser = argparse.ArgumentParser(
    description="批量筛选 *.ratio.txt 文件并重写ID (两列输出)，仅输出 Pe/Span >= 1 的记录"
)
parser.add_argument(
    "-i", "--input", required=True,
    help="主目录路径（包含子文件夹，每个子文件夹里有 *_bowtie1.ratio.txt 文件）"
)
parser.add_argument(
    "-o", "--output", default="filtered_ids.txt",
    help="符合条件的输出文件 (默认: filtered_ids.txt)"
)
parser.add_argument(
    "-n", "--notpassed", default="not_passed_ids.txt",
    help="未满足条件的输出文件 (默认: not_passed_ids.txt)"
)
args = parser.parse_args()

main_dir = args.input
output_file_passed = args.output
output_file_failed = args.notpassed


def safe_float(value):
    """Safely convert a string to a float"""
    if str(value).strip().upper() in {"NA", "", "N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# =====================================================
# 扫描目录并筛选数据
# =====================================================
print("\n=== 开始筛选 >=1 的记录 ===")

passed_rows = []
failed_rows = []

for subdir in os.listdir(main_dir):
    sub_path = os.path.join(main_dir, subdir)
    if not os.path.isdir(sub_path):
        continue

    ratio_files = glob.glob(os.path.join(sub_path, "*_bowtie1.ratio.txt"))
    if not ratio_files:
        print(f"In {subdir} no *_bowtie1.ratio.txt files found, skipping.")
        continue

    ratio_file = ratio_files[0]
    print(f"Processing {ratio_file} ...")
    with open(ratio_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required_cols = {"Pe_R", "Span_R", "Pe_F", "Span_F", "ID"}
        if not required_cols.issubset(reader.fieldnames):
            print(f"File {ratio_file} missing columns: {required_cols - set(reader.fieldnames)}, skipping.")
            continue

        for row in reader:
            Pe_R = safe_float(row.get("Pe_R", ""))
            Span_R = safe_float(row.get("Span_R", ""))
            Pe_F = safe_float(row.get("Pe_F", ""))
            Span_F = safe_float(row.get("Span_F", ""))
            old_id = row.get("ID", "")

            # 重写 ID 为 “子文件夹:原ID尾部”
            if ":" in old_id:
                prefix, tail = old_id.split(":", 1)
                new_id = f"{subdir}:{tail}"
            else:
                prefix = ""
                new_id = f"{subdir}:{old_id}"

            passed = (
                Pe_R is not None and Span_R is not None and (Pe_R >= 1 or Span_R >= 1)
            )

            record = (new_id, prefix, Pe_F, Pe_R, Span_F, Span_R)
            if passed:
                passed_rows.append(record)
            else:
                failed_rows.append(record)


# =====================================================
# 写出结果文件
# =====================================================
print("\n=== Writing output files ===")

with open(output_file_passed, "w", encoding="utf-8") as f1, open(output_file_failed, "w", encoding="utf-8") as f2:
    header = "ID\tSample\tPe_F\tPe_R\tSpan_F\tSpan_R\n"
    f1.write(header)
    f2.write(header)
    
    for r in passed_rows:
        f1.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]}\n")
    for r in failed_rows:
        f2.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]}\n")

print("\nProcessing complete!")
print(f"→ Passed output: {output_file_passed}")
print(f"→ Failed output: {output_file_failed}")