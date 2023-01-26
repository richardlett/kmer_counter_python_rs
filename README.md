
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
To tune for current CPU

```
maturin develop --release -- -C target-cpu=native
```

Put in a (optionally gzipped) fasta. indices are corresponding (contig_names[i] is name for and  fivemers[i] is fivemer counts)
```
import kmer_counter
import numpy as np

tmp  = kmer_counter.find_nMer_distributions("/pscratch/sd/r/richardl/skin/contigs.fna.gz")

(l4n1mers, fivemers,fourmers,threemers,twomers,onemers,mer10,mer9,mer8,mer7,mer6, contig_names) = (np.reshape(tmp[0], (-1,136)),np.reshape(tmp[1], (-1,512)),np.reshape(tmp[2], (-1,136)),np.reshape(tmp[3], (-1,32)),np.reshape(tmp[4], (-1,10)), np.reshape(tmp[5], (-1,2)), np.reshape(tmp[6], (-1,528)), np.reshape(tmp[7], (-1,256)),np.reshape(tmp[8], (-1,136)),np.reshape(tmp[9], (-1,64)),np.reshape(tmp[10], (-1,36)), tmp[11])
```

Note: I turned off multithreaded file io because perlmutter was giving me bus errors for no reason. I blame Perlmutter. I also skip contigs less than 1000 in length, as it's just wasted cpu cycles for me

I currently don't return length of contigs, you can do that, but I can return 12 things max into python, and didn't need it

at line 861 in lib.rs, change 

```
        (
            pre_l4n1mers.into_pyarray(py),
            pre_5mers.into_pyarray(py),
            pre_4mers.into_pyarray(py),
            pre_3mers.into_pyarray(py),
            pre_2mers.into_pyarray(py),
            pre_1mers.into_pyarray(py),
            pre_10mers.into_pyarray(py),
            pre_9mers.into_pyarray(py),
            pre_8mers.into_pyarray(py),
            pre_7mers.into_pyarray(py),
            pre_6mers.into_pyarray(py),
            contig_names
        )
```

and change the function signature at line 704 to `usize` instead of `f32` for whatever one you replace I beleive.

Edit l4n1mers are bugged. I dont use those anymore
