import os
import argparse
import csv
import glob


def safe_float(value):
    """Safely convert a string to a float; return None for NA/empty."""
    if value is None:
        return None
    s = str(value).strip()
    if s.upper() in {"NA", "N/A", ""}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch filter *.ratio.txt files in subfolders and write matching rows to one output file. "
            "Rule: keep rows where Pe_R >= 1 OR Span_R >= 1."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Main directory containing subfolders with *.ratio.txt files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="passed.ratio.rows.txt",
        help="Output file for rows that pass the filter (default: passed.ratio.rows.txt).",
    )
    parser.add_argument(
        "-n",
        "--notpassed",
        default="failed.ratio.rows.txt",
        help="Output file for rows that fail the filter (default: failed.ratio.rows.txt).",
    )
    args = parser.parse_args()

    main_dir = args.input
    out_pass = args.output
    out_fail = args.notpassed

    required_cols = {
        "ID",
        "Pe_F",
        "Pe_R",
        "Pe_ratio",
        "Span_F",
        "Span_R",
        "Span_ratio",
    }

    passed_rows = []
    failed_rows = []
    header = None

    for subdir in os.listdir(main_dir):
        sub_path = os.path.join(main_dir, subdir)
        if not os.path.isdir(sub_path):
            continue

        ratio_files = glob.glob(os.path.join(sub_path, "*.ratio.txt"))
        if not ratio_files:
            print(f"In {subdir}: no *.ratio.txt files found, skipping.")
            continue

        ratio_file = ratio_files[0]
        print(f"Processing {ratio_file} ...")

        with open(ratio_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            missing = required_cols - set(fieldnames)
            if missing:
                print(f"File {ratio_file} is missing columns: {missing}, skipping.")
                continue

            if header is None:
                header = fieldnames

            for row in reader:
                pe_r = safe_float(row.get("Pe_R"))
                span_r = safe_float(row.get("Span_R"))

                passed = (
                    pe_r is not None
                    and span_r is not None
                    and (pe_r >= 1 or span_r >= 1)
                )

                if passed:
                    passed_rows.append(row)
                else:
                    failed_rows.append(row)

    if header is None:
        raise SystemExit("No valid *.ratio.txt files were found (or all were missing required columns).")

    def write_rows(path, rows):
        with open(path, "w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=header, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in header})

    write_rows(out_pass, passed_rows)
    write_rows(out_fail, failed_rows)

    print("\nProcessing complete!")
    print(f"→ Passed rows: {len(passed_rows)} written to {out_pass}")
    print(f"→ Failed rows: {len(failed_rows)} written to {out_fail}")


if __name__ == "__main__":
    main()