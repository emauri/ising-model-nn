Simple one layer feed forward neural network for detecting phases in the Ising model, implemented using *armadillo C++ library*.

The zip file *data.zip* contains 4 directories:
  * *dataList*, containing 3 files with a list of all the trining, validation and test data files respectively
  * *training*, containing the data files for training
  * *validation*, containing the data files for validation
  * *test*, containing the data files for testing

A makefile is provided to compile the program. Currently it is set to use Mac OsX *-framework Accelerate*, that provides a multithreading implementation of BLAS.
To run the program on Linux with the multithreaded *openBlas* library installed, uncomment the makefile lines linking to the *openBlas* library, you may also need to chabge the path to match the one on your machine.
