
# kmer_counter_python_rs

# Intro
This github repo is the kmer counting module for GenomeFace. It has a few  functions

1. Count kmers from an arbitrarily (optionally gzipped) fasta file, returning numpy arrays of l1 normalzied kmer counts, along with contig names (1-5,6-10 degenerate RYmers).
2. Create a "database" datastructure of genomes which contigs can be sampled from for training the compositional neural network. Genomes are loaded in. When sample is called on the database datastructure, a similar to point #1, a bunch of kmers are returned in numpy format, along with an integer indicating which genome it came from (used for ground truth to train the neural network with.
3. It also has an accesorry function for rewriting contigs from an assembly into individual bins.
4. Do it all fast. Written in Rust and multithreaded.  Thge training database releases the Python Global Interpreter Lock, allowing tensorflow to resume execution. This allows tensorflow to train on batch N with the GPUs while we are generating training data for batches N+1, N+2 with the CPU.

# Build and install instructions
If you don't have rust, install it with (yes, this curl is offical installer)
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

then Install maturin, a toolchain for rust/python integration package

```
mamba install -c conda-forge maturin
# or 
conda install -c conda-forge maturin
# or
pip install maturin

```

To build and install this python package to your current python environment,  `cd` into this repo, and 
```
maturin develop --release
```
To tune for current system's CPU (this may break system compatibility!)

```
maturin develop --release -- -C target-cpu=native
```

note, you can set CARGO_HOME enviorment variable to scratch to save space when downloading dependencies

```
mkdir $SCRATCH/cargo
CARGO_HOME=$SCRATCH/cargo maturin develop --release -- -C target-cpu=native
```

# Dev Usage
## Kmer Counting from fasta

The numpy arrays are not shaped correctly upon return. (I should probably add something cleaner, but this is more of an internal library for my own use)

Anyway, we just need to resize them for a 

```python
import kmer_counter
import numpy as np
input_file = "filepath.fasta.gz"
min_contig_len = 1500 # shorter will be ignored
aaq = kmer_counter.find_nMer_distributions(input_file, min_contig_length)
# a bunch of 1d numpy arrays are returned. We reshape them to appropriate size.
# The actual indexing of which canonical kmer is what index is arbitrary (you can check the code),
# but the corresponding kmers are: 5, 4, 3, 2, 1, Then 10, 9, 8 ,7,5 for  degenerate RY-mers.
# (for example, notice there are 136 canonical 4-mers.)
inpts = [np.reshape(aaq[i], (-1, size)) for i, size in enumerate([512, 136, 32, 10, 2, 528, 256, 136, 64, 36], start=1)]

# Usage: inputs[0][n] gives 5-mer count array for nth contig in fasta.

contig_names = np.asarray(aaq[-1])

contig_lens = np.asarray(aaq[0])
# Usage: contig_names[n], contig_lens[n] gives name, length of nth contig in fasta
```

## Training dataset generator:
Put in a list of fasta files. Each will be considered a training class. The second number is min contig size for loading into database. This reduces memory requirements.
```python
from kmer_counter import FastaDataBase
db = FastaDataBase(['genome1.fa.gz','genome1.fa.gz`],1000)
# only sample greater than 2000 len, 1048576 // 4 samples are returned.
numpy_arrays = db.sample(1048576 // 4, 2000)
Usage as above
````

more complicated example that returns a tensorflow dataset generator:

```python

import tensorflow as tf

def generate_data(file_list):
    db = FastaDataBase(file_list, 2000)
    n_classes = len(file_list)
    print("Reading done")

    def data_generator():
        while True:
            aaq = db.sample(1048576 // 4, 2000)
            m = tf.one_hot(aaq[-1], n_classes + 1)
            input_tensors = tuple(tf.reshape(aaq[i], (-1, shape)) for i, shape in enumerate([512, 136, 32, 10, 2, 528, 256, 136, 64, 36]))
            yield input_tensors, m

    input_signature = tuple(
        tf.TensorSpec(shape=(1048576 // 4, shape), dtype=tf.float32) for shape in [512, 136, 32, 10, 2, 528, 256, 136, 64, 36]
    )

    data = tf.data.Dataset.from_generator(data_generator, output_signature=(input_signature, tf.TensorSpec(shape=(1048576 // 4, n_classes + 1), dtype=tf.float32))).prefetch(tf.data.AUTOTUNE).unbatch().batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    return data, n_classes
```

## Fasta bin writer.

Takes the 
1.  Original fasta file of assembly, contigs will be retrieved from this based on names fed in.
2.  Two numpy arrays. One which contains integer labels (two contigs are in same bin if they have same integer label). The other of corresponding contig name.
3.  output Folder path to write output bins. Output Folder should not exist, it will be created

example
```python
import kmer_counter
input_fasta = "contigs.fa.gz"
output_folder_path = "genomeface_bins/"
contig_names = ["contig1", "contig2", "contig3, "contig4", "contig5", "contig6"]
contig_labels = [0, 1, 1, 3, 2, 2] # there are 4 bins. contig 1 belongs to bin 0. contig2  belongs to bin 1, etc.

bases_binned = kmer_counter.write_fasta_bins(contig_names, contig_labels, input_fasta,  output_folder_path)
print("There were", bases_binned,"bases binned.")

```
