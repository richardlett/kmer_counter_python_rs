# kmer_counter_python_rs

## Intro
This GitHub repository contains the k-mer counting module for GenomeFace. It offers several functions:

1. Count k-mers from an arbitrary (optionally gzipped) fasta file. It returns numpy arrays of L1 normalized k-mer counts along with the contig names (ranging from 1-5 to 6-10 for degenerate RY-mers).
2. Construct a "database" data structure from genomes, allowing contigs to be sampled. This aids in training the compositional neural network. Once the genomes are loaded, the `sample` function returns L1 normalized k-mers in numpy format with an integer indicating the genome source, which is then used as ground truth for neural network training.
3. Provide an accessory function for rewriting contigs from an assembly into individual bins.
4. Ensure high performance by being written in Rust and leveraging multithreading. The training database releases the Python Global Interpreter Lock, letting TensorFlow execute. This means TensorFlow can train on batch N using the GPUs while concurrently generating training data for batches N+1, N+2, etc. on the CPU.

## Build and Install Instructions
If you don't have Rust installed, use the official installer:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Next, install maturin, a toolchain for integrating Rust and Python:

```bash
mamba install -c conda-forge maturin
# or 
conda install -c conda-forge maturin
# or
pip install maturin
```

To build and install this Python package in your current Python environment, navigate (`cd`) to this repository and run:

```bash
maturin develop --release
```

To optimize for your current system's CPU (be cautious as this might reduce compatibility):

```bash
maturin develop --release -- -C target-cpu=native
```

If you wish to save space when downloading dependencies, you can set the CARGO_HOME environment variable to scratch:

```bash
mkdir $SCRATCH/cargo
CARGO_HOME=$SCRATCH/cargo maturin develop --release -- -C target-cpu=native
```

Note: `maturin build --release` will build a python '.whl' package for distribution, which can be installed with `pip install filename.whl`. It will be placed in a subdirectory of `repo/target/` (but I forgot where).

The built package is statically linked, and has no dependencies (besides numpy).

## Dev Usage
### K-mer Counting from Fasta
The numpy arrays returned are not shaped correctly by default. Although I hope to have a cleaner API later, for now, you can resize them as follows:

```python
import kmer_counter
import numpy as np

input_file = "filepath.fasta.gz"
min_contig_len = 1500  # Contigs shorter than this will be ignored
aaq = kmer_counter.find_nMer_distributions(input_file, min_contig_len)

# Reshape the returned 1D numpy arrays to their appropriate sizes.
# Refer to the code to determine which canonical k-mer corresponds to each index.
inpts = [np.reshape(aaq[i], (-1, size)) for i, size in enumerate([512, 136, 32, 10, 2, 528, 256, 136, 64, 36], start=1)]

# Example: inpts[1][n] provides the 4-mer count array (of size 136) for the nth contig in the fasta.
contig_names = np.asarray(aaq[-1])
contig_lens = np.asarray(aaq[0])
```
The first five indexes correspond to canonical (i.e. rev comp seen as equivalent) 5,4,3,2,1 mer frequencies,then 10,9,8,7,6 mers canonical RY-mer frequencies

### Training Dataset Generator:
Supply a list of fasta files. Each file will be considered as a training class. The second number indicates the minimum contig size to load into the database to manage memory requirements.

```python
from kmer_counter import FastaDataBase

db = FastaDataBase(['genome1.fa.gz','genome2.fa.gz'], 1000)
# Sample contigs greater than 2000 in length. It returns 1048576 // 4 samples.
numpy_arrays = db.sample(1048576 // 4, 2000)
```

For a more advanced example that yields a TensorFlow dataset generator:

```python
import tensorflow as tf
from kmer_counter import FastaDataBase

def generate_data(file_list):
    db = FastaDataBase(file_list, 2000)
    n_classes = len(file_list)
    print("Reading done")

    def data_generator():
        while True:
            aaq = db.sample(1048576 // 4, 2000)
            # Convert the class label (integer) to one-hot encoding for categorical cross-entropy
            m = tf.one_hot(aaq[-1], n_classes + 1)
            input_tensors = tuple(tf.reshape(aaq[i], (-1, shape)) for i, shape in enumerate([512, 136, 32, 10, 2, 528, 256, 136, 64, 36]))
            yield input_tensors, m

    input_signature = tuple(
        tf.TensorSpec(shape=(1048576 // 4, shape), dtype=tf.float32) for shape in [512, 136, 32, 10, 2, 528, 256, 136, 64, 36]
    )

    data = tf.data.Dataset.from_generator(data_generator, output_signature=(input_signature, tf.TensorSpec(shape=(1048576 // 4, n_classes + 1), dtype=tf.float32))).prefetch(tf.data.AUTOTUNE).unbatch().batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    return data, n_classes
```

### Fasta Bin Writer:

This function:

1. Reads the original fasta file of the assembly.
2. Accepts two lists: one containing integer labels (indicating that two contigs belong to the same bin if they share the same integer label), and the other listing the corresponding contig names.
3. Specifies an output folder path to store the output bins. Ensure the output folder doesn't already exist, as it will be created.

Here's an example:

```python
import kmer_counter

input_fasta = "contigs.fa.gz"
output_folder_path = "genomeface_bins/"
contig_names = ["contig1", "contig2", "contig3", "contig4", "contig5", "contig6"]
contig_labels = [0, 1, 1, 3, 2, 2]  # Contig labels for bins. For example, contig1 belongs to bin 0, contig2 to bin 1, and so on.

bases_binned = kmer_counter.write_fasta_bins(contig_names, contig_labels, input_fasta, output_folder_path)
print("There were", bases_binned, "bases binned.")
```
