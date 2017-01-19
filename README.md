One hidden layer feed forward neural network for Ising model phase detection using Armadillo C++ library with LAPACK & OpenBLAS packages.

Result on full training (33000), validation (5000) and test (5000) data:
 - Without OpenBLAS
 - Graph extracted from raw data: outputCartesius.txt
 
![Data Distribution Scheme Result from Cartesius](/Data Distribution Scheme without OpenBLAS.png?raw=true "Data Distribution Scheme without OpenBLAS")

Result on full training (33000), validation (5000) and test (5000) data:
 - Single thread OpenBLAS (Multithread OpenBLAS doesn't have effect within BSP function)
 - Graph extracted from raw data: outputCartesiusOpenBlas.txt
 
 
![Data Distribution Scheme Result from Cartesius](/Data Distribution Scheme with OpenBLAS.png?raw=true "Data Distribution Scheme with OpenBLAS")
