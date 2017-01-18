One hidden layer feed forward neural network for Ising model phase detection using Armadillo C++ library with LAPACK & OpenBLAS packages.

Result on full training (33000), validation (5000) and test (5000) data:
 - Single thread OpenBLAS (Multithread OpenBLAS doesn't have effect within BSP function)
 - Batch script: jobscript.sh running on Cartesius with 1 to 24 cores range
 - Graph extracted from raw data: outputCartesius.txt
 
![Data Distribution Scheme Result from Cartesius](/Data Distribution Scheme.png?raw=true "Data Distribution Scheme Result from Cartesius")
