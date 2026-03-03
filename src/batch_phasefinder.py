import argparse
import sys
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_cmd(cmd: str) -> int:
    """Run shell command and return its exit code."""
    print(f"[CMD] {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def extract_id(fasta_path: Path, sep: str = "_", field: int = 0) -> str:
    """
    Extract sample ID from fasta filename.
    Default: take the first token split by '_' (field=0).
    """
    parts = fasta_path.name.split(sep)
    if field < 0 or field >= len(parts):
        raise ValueError(
            f"Cannot extract ID: field={field} out of range for filename={fasta_path.name} split by '{sep}'"
        )
    return parts[field]


def find_reads(
    reads_dir: Path,
    sample_id: str,
    reads_prefix: str,
    r1_suffix: str,
    r2_suffix: str,
):
    """
    Find read1/read2 files by exact path; if not exists, try glob as fallback.
    Pattern:
      R1: {sample_id}{reads_prefix}{r1_suffix}
      R2: {sample_id}{reads_prefix}{r2_suffix}
    """
    r1 = reads_dir / f"{sample_id}{reads_prefix}{r1_suffix}"
    r2 = reads_dir / f"{sample_id}{reads_prefix}{r2_suffix}"

    if r1.exists() and r2.exists():
        return r1, r2

    # fallback: allow extra stuff between prefix and suffix
    r1_glob = list(reads_dir.glob(f"{sample_id}{reads_prefix}*{r1_suffix}"))
    r2_glob = list(reads_dir.glob(f"{sample_id}{reads_prefix}*{r2_suffix}"))
    if r1_glob and r2_glob:
        return r1_glob[0], r2_glob[0]

    return None, None


def phasefinder(
    file_path: Path,
    output_dir: Path,
    reads_dir: Path,
    reads_prefix: str,
    r1_suffix: str,
    r2_suffix: str,
    phasefinder_py: Path,
    id_sep: str,
    id_field: int,
):
    """
    Run PhaseFinder on one fasta file:
      locate -> create -> ratio
    """
    logs = []

    sample_id = extract_id(file_path, sep=id_sep, field=id_field)
    print(f"Processing {sample_id}", flush=True)

    reads1_gz, reads2_gz = find_reads(
        reads_dir=reads_dir,
        sample_id=sample_id,
        reads_prefix=reads_prefix,
        r1_suffix=r1_suffix,
        r2_suffix=r2_suffix,
    )

    logs.append(f"Using genome: {file_path}")

    if reads1_gz is None or reads2_gz is None:
        return False, (
            f"Missing reads for {sample_id}\n"
            f"  reads_dir={reads_dir}\n"
            f"  tried exact: {reads_dir}/{sample_id}{reads_prefix}{r1_suffix}\n"
            f"              {reads_dir}/{sample_id}{reads_prefix}{r2_suffix}\n"
            f"  and glob fallback with '*{r1_suffix}' / '*{r2_suffix}'"
        )

    logs.append(f"Reads: {reads1_gz}, {reads2_gz}")

    output_sample_dir = output_dir / sample_id
    output_sample_dir.mkdir(parents=True, exist_ok=True)

    tab_path = output_sample_dir / f"{sample_id}.einverted.tab"
    id_fasta = output_sample_dir / f"{sample_id}.ID.fasta"
    out_prefix = output_sample_dir / f"{sample_id}"

    try:
        py = sys.executable  # use the same python running this script
        commands = [
            f"{py} {phasefinder_py} locate "
            f"-f {file_path} -t {tab_path} -g 15 85 -p",

            f"{py} {phasefinder_py} create "
            f"-f {file_path} -t {tab_path} "
            f"-s 1000 -i {id_fasta}",

            f"{py} {phasefinder_py} ratio "
            f"-i {id_fasta} "
            f"-1 {reads1_gz} -2 {reads2_gz} -o {out_prefix}",
        ]

        for idx, cmd in enumerate(commands, 1):
            print(f"Step {idx}/3: {cmd}", flush=True)
            ret = run_cmd(cmd)
            if ret != 0:
                return False, f"Step {idx} failed for {sample_id}"

        logs.append(f"PhaseFinder completed successfully for {sample_id}")
        return True, "\n".join(logs)

    except Exception as e:
        return False, f"{sample_id}: Error {e}"


def parallel_phasefinder(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in args.input_dir.iterdir() if f.is_file() and f.suffix == ".fasta"]
    print(f"Found {len(files)} fasta files to process.", flush=True)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                phasefinder,
                f,
                args.output_dir,
                args.reads_dir,
                args.reads_prefix,
                args.r1_suffix,
                args.r2_suffix,
                args.phasefinder,
                args.id_sep,
                args.id_field,
            ): f
            for f in files
        }

        for future in as_completed(futures):
            fasta = futures[future]
            try:
                ok, msg = future.result()
                print(msg, flush=True)
                print("=" * 70)
                print(("Finished: " if ok else "Failed: ") + fasta.name, flush=True)
            except Exception as e:
                print(f"Error processing {fasta.name}: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Batch run PhaseFinder on fasta files (locate/create/ratio) with configurable reads naming."
    )
    parser.add_argument("-i", "--input_dir", required=True, type=Path, help="Input directory of fasta files")
    parser.add_argument("-o", "--output_dir", required=True, type=Path, help="Output directory for results")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of parallel threads")

    # PhaseFinder
    parser.add_argument(
        "-pf",
        "--phasefinder",
        type=Path,
        help="Path to PhaseFinder.py",
    )

    # How to get sample ID from fasta filename
    parser.add_argument("--id_sep", type=str, default="_", help="Separator used to split fasta filename for ID extraction")
    parser.add_argument(
        "--id_field",
        type=int,
        default=0,
        help="Which field to take after splitting by --id_sep (0-based). Default 0 => first token.",
    )

    # Reads location & naming
    parser.add_argument(
        "-r",
        "--reads_dir",
        type=Path,
        required=True,
        help="Directory containing reads files",
    )
    parser.add_argument(
        "-rp",
        "--reads_prefix",
        type=str,
        default="",
        help="String inserted between ID and read suffix (e.g. '_clean' for ID_clean_1.fastq.gz). Can be empty.",
    )
    parser.add_argument("--r1_suffix", type=str, default="_1.fastq.gz", help="Suffix for read1 filename")
    parser.add_argument("--r2_suffix", type=str, default="_2.fastq.gz", help="Suffix for read2 filename")

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"--input_dir not found: {args.input_dir}")
    if not args.reads_dir.exists():
        raise SystemExit(f"--reads_dir not found: {args.reads_dir}")
    if not args.phasefinder.exists():
        raise SystemExit(f"--phasefinder not found: {args.phasefinder}")

    parallel_phasefinder(args)


if __name__ == "__main__":
    main()