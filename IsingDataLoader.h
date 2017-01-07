//Header for a simple IsingDataLoader class, to be improved later
#ifndef ISINGDATALOADER_H
#define ISINGDATALOADER_H
#include<armadillo>

class IsingDataLoader {

  //Class members
  //----------------------------------------------------------------------------
private:

  //data structure to load the data, is a field of field of vectors. Each field
  //represent one training data, each containing a input configuration and a
  //output label.
  arma::field< arma::field<arma::fvec> > set;

  //Public methods
  //----------------------------------------------------------------------------
public:

  //Load the data set.
  void loadData(uint32_t numberOfFiles, const char * filenames[]);

  //Getter for the data set. Return a pointer to the loaded dat set.
  arma::field< arma::field<arma::fvec> > * getDataSet();



  //Private methods
  //----------------------------------------------------------------------------
private:

  //initialize one element of the set with input data and output label
  void setData(arma::field<arma::fvec> & data, const char * filename);

};

#endif
