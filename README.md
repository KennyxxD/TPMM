# TPMM
TPMM is designed to identify invertons from metagenomic data in the same environment, especially the low-depth ones, based on the raw results of PhaseFinder. It helps define a high-confidence inverton set and is able to detect more invertons when the depth of reads is low. 
## Prepare the environment
Here, we recommend to prepare two seperate environments, one for preprocessing data using PhaseFinder, one for TPMM.
* To install  PhaseFinder, please refer to [PhaseFinder](https://github.com/XiaofangJ/PhaseFinder)
* To install TPMM
```bash
git clone https://github.com/KennyxxD/TPMM.git
conda env create -n TPMM -f environment.yml
conda activate TPMM
```
## Preprocess the data
Assuming that you have many data samples including contigs and reads from the same environment, please put the contigs(fasta) and reads(.fq) in two folders. Then you need to run
```bash
conda activate PhaseFinder
```
Then you need to run `src/batch_phasefinder.py` based on the usage instructions below.
```
usage: batch_phasefinder.py [-h] -i INPUT_DIR -o OUTPUT_DIR [-t THREADS] [-pf PHASEFINDER] [--id_sep ID_SEP] [--id_field ID_FIELD] -r READS_DIR [-rp READS_PREFIX] [--r1_suffix R1_SUFFIX]
                                   [--r2_suffix R2_SUFFIX]

Batch run PhaseFinder on fasta files (locate/create/ratio) with configurable reads naming.

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Input directory of fasta files
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory for results
  -t THREADS, --threads THREADS
                        Number of parallel threads
  -pf PHASEFINDER, --phasefinder PHASEFINDER
                        Path to PhaseFinder.py
  --id_sep ID_SEP       Separator used to split fasta filename for ID extraction
  --id_field ID_FIELD   Which field to take after splitting by --id_sep (0-based). Default 0 => first token.
  -r READS_DIR, --reads_dir READS_DIR
                        Directory containing reads files
  -rp READS_PREFIX, --reads_prefix READS_PREFIX
                        String inserted between ID and read suffix (e.g. '_clean' for ID_clean_1.fq). Can be empty.
  --r1_suffix R1_SUFFIX
                        Suffix for read1 filename
  --r2_suffix R2_SUFFIX
                        Suffix for read2 filename
```
In order to help you better understand the usage, we make an example:
If you put the contigs into the folder `data/contigs/` and the name of the files is like `CONTIGX.fasta`. Meanwhile, you put the reads into the folder `data/reads/`, and the names of the file are like `CONTIGX_clean_1.fq` and `CONTIGX_clean_2.fq`. Then you can run
```
python batch_phasefinder.py -i data/contigs -o <your output dir> -t <threads> -pf <path to PhaseFinder.py> --id_sep _ --id_field 0 -r data/reads -rp _clean
```
## Get candidates for TPMM
You need to run `src/get_candidates.py`, the instructions of usage is below:
```
usage: get_candidates.py [-h] -i INPUT [-o OUTPUT] [-n NOTPASSED]

Batch filter *.ratio.txt files in subfolders and write matching rows to one output file. Rule: keep rows where Pe_R >= 1 OR Span_R >= 1.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Main directory containing subfolders with *.ratio.txt files.
  -o OUTPUT, --output OUTPUT
                        Output file for rows that pass the filter (default: passed.ratio.rows.txt).
  -n NOTPASSED, --notpassed NOTPASSED
                        Output file for rows that fail the filter (default: failed.ratio.rows.txt).
```
Here, the input folder is the output folder of last step and you will a txt file for all the candidates with Pe_R >= 1 or Span_R >= 1.
## Estimation and obtain results
The next step is to apply TPMM to get the posterior and use Bayesian FDR to get hard calling. 
```bash
conda activate TPMM
```
Then you need to run `src/TPMM.py`
```
usage: TPMM.py [-h] --input INPUT [--sep SEP] [--min_coverage MIN_COVERAGE] [--q_low Q_LOW] [--q_high Q_HIGH] [--max_iter MAX_ITER] [--tol TOL] [--restarts RESTARTS] [--seed SEED]
                    [--posterior_min POSTERIOR_MIN] [--posterior_col {posterior_true,posterior_non_noise,posterior_high,posterior_low,posterior_noise}] [--use_bfdr] [--bfdr_alpha BFDR_ALPHA]
                    [--outdir OUTDIR] [--prefix PREFIX]

Three-component Binomial EM on high-N subset, score all; report hits.

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input TSV path (e.g., gut1_pos.txt)
  --sep SEP             Separator (default: tab)
  --min_coverage MIN_COVERAGE
                        Keep rows with N>=min_coverage (default: 8)
  --q_low Q_LOW         Lower quantile for fit subset (default: 0.90)
  --q_high Q_HIGH       Upper quantile for fit subset (default: 0.99)
  --max_iter MAX_ITER   EM max iterations (default: 500)
  --tol TOL             EM tolerance on loglike (default: 1e-6)
  --restarts RESTARTS   EM restarts (default: 25)
  --seed SEED           Random seed (default: 0)
  --posterior_min POSTERIOR_MIN
                        Hit filter: posterior_true>= (default: 0.90). Now posterior_true=posterior_non_noise.
  --posterior_col {posterior_true,posterior_non_noise,posterior_high,posterior_low,posterior_noise}
                        Which posterior column used in hit filter & grid summaries (default: posterior_true=non_noise)
  --use_bfdr            Use posterior-based Bayesian FDR selection (cumulative lfdr = posterior_noise).
  --bfdr_alpha BFDR_ALPHA
                        Target Bayesian FDR alpha for --use_bfdr (default: 0.05).
  --outdir OUTDIR       Output directory
  --prefix PREFIX       Output prefix
```
After that, you will get the results for all the data points in the `.em.tsv` file and the positive results in the `.hits.tsv` file. The detailed posterior and BFDR value are provided.
