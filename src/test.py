from kmer_counter import FastaDataBase
import os
import gzip
import time

base_dir = "/pscratch/sd/r/richardl/bacteria/"
files = [base_dir +file for file in  os.listdir(base_dir) if len(file) >= 3 and (file[-3:] == "fna" or  file[-3:] == ".gz")   ]

print(len(files))
tic = time.perf_counter()
db = FastaDataBase(files[:10],10001)
toc = time.perf_counter()
print(f"Fasta load time: {toc - tic:0.4f} seconds")

tic = time.perf_counter()

qes = db.sample(10000, 10000)
toc = time.perf_counter()

print(len(qes))
print(type(qes))
fss
print(f"Fasta sample time: {toc - tic:0.4f} seconds")

from Bio import SeqIO



fasta_sequences = SeqIO.parse(gzip.open("/pscratch/sd/r/richardl/bacteria/GCA_000006685.1_ASM668v1_genomic.fna.gz","rt"),'fasta')
seqs = []
for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    seqs.append(sequence)



for idx, contig in enumerate(seqs):
    test_contig = db.get_contig(0,idx)
    assert(contig == test_contig)

print("idx correct", idx+1)