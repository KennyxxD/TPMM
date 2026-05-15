# Step 1
trimmomatic PE -threads <threads> \
  <input_reads_1> <input_reads_2> \
  sample_R1_paired.fq.gz sample_R1_unpaired.fq.gz \
  sample_R2_paired.fq.gz sample_R2_unpaired.fq.gz \
  ILLUMINACLIP:TruSeq3-PE-IDT_dual_index.fa:2:30:10:2:true \
  HEADCROP:1 \
  LEADING:20 TRAILING:20 \
  SLIDINGWINDOW:4:20 \
  AVGQUAL:20 \
  MINLEN:60

# Step 2
bowtie2 -N 1 -p <threads> \
  -x <host_bowtie2_index_prefix> \
  -1 sample_R1_paired.fq.gz \
  -2 sample_R2_paired.fq.gz \
  --un-conc-gz sample_nonhost_R%.fq.gz \
  -S /dev/null

# Step 3
megahit -t <threads> \
  -1 sample_nonhost_R1.fq.gz \
  -2 sample_nonhost_R2.fq.gz \
  -o sample_megahit_out