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
Assuming that you have many data samples including contigs and reads from the same environment, please put the contigs(fasta) and reads(fastq.gz or .gz) in two folders.
```bash
python batch_phasefinder_github.py -h 
usage: batch_phasefinder_github.py [-h] -i INPUT_DIR -o OUTPUT_DIR [-t THREADS] [-pf PHASEFINDER] [--id_sep ID_SEP] [--id_field ID_FIELD] -r READS_DIR [-rp READS_PREFIX] [--r1_suffix R1_SUFFIX]
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
                        String inserted between ID and read suffix (e.g. '_clean' for ID_clean_1.fastq.gz). Can be empty.
  --r1_suffix R1_SUFFIX
                        Suffix for read1 filename
  --r2_suffix R2_SUFFIX
                        Suffix for read2 filename
```
