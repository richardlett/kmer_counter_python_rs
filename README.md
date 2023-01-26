
If you don't have rust, install it (yes, this curl is offical installer)
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

then Install maturin, the rust/python integration package

```
conda install -c conda-forge maturin
```
or

```
pip install maturin
```

To build and install the package to your current python enviorment,
```
maturin develop --release
```
Put in a (optionally gzipped) fasta. indices are corresponding (contig_names[i] is name for and  fivemers[i] is fiver counts)
```
tmp  = kmer_counter.find_nMer_distributions("/pscratch/sd/r/richardl/skin/contigs.fna.gz")
(l4n1mers, fivemers,fourmers,threemers,twomers,onemers,mer10,mer9,mer8,mer7,mer6, contig_names) = (np.reshape(tmp[0], (-1,136)),np.reshape(tmp[1], (-1,512)),np.reshape(tmp[2], (-1,136)),np.reshape(tmp[3], (-1,32)),np.reshape(tmp[4], (-1,10)), np.reshape(tmp[5], (-1,2)), np.reshape(tmp[6], (-1,528)), np.reshape(tmp[7], (-1,256)),np.reshape(tmp[8], (-1,136)),np.reshape(tmp[9], (-1,64)),np.reshape(tmp[10], (-1,36)), tmp[11])
```