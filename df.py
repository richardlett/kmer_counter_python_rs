import time
import numpy as np
#import tensorflow as tf
start = time.time()
import kmer_counter
print("kmer count module import time: ",time.time() - start)


start =time.time()
res = kmer_counter.find_nMer_distributions("/pscratch/sd/r/richardl/contigs_air.fna.gz")
print("read / count time: ",time.time() - start)
print(res[0][527])

mer5_tensor =  np.load("/pscratch/sd/r/richardl/tensor4.npz.npy")
l4n1_tensor =  (np.load("/pscratch/sd/r/richardl/tensor0.npz.npy"))

for i in range(136):
    print (res[1][i], l4n1_tensor[0,i])
