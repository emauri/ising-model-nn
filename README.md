Simple one layer feed forward neural network for detecting phases in the Ising model, implemented using *armadillo C++ library*.

A makefile is provided to compile the algorithm. Currently it is set to use Mac OsX *-framework Accelerate*, that provides a multithreading implementation of BLAS.
To run the program on Linux with the multithreaded *openBlas* library installed, uncomment the makefile lines linking to the *openBlas* library, you may also need to chabge the path to match the one on your machine.
