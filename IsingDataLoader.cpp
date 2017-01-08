//implementation of the IsingDataLoader class

//include class declaration
#include "IsingDataLoader.h"
#include<iostream>

using namespace arma;

//getter
field< field<fvec> > * IsingDataLoader::getDataSet() {
  return &(set);
}

//Read data from one file and initialize one elemnt of set, i.e. it initializes
//two vectors, one with the input configuration and one with the corresponding ouput label.
void IsingDataLoader::setData(field<fvec> & data, const char * filename) {

  //set data size to 2 (input configuration, output label)
  data.set_size(2);

  //load the data in the file, the first line contain the size of the Ising lattice,
  //the second line contains the temperature of the system. From the third line
  //onwards the file contains the Ising spin configuration
  fvec fileData;
  fileData.load(filename);

  //Set the output label
  //----------------------------------------------------------------------------
  //If the temperature is less than 2.69 it is below the critical temperature: ouput = {0,1},
  //otherwise it is above: output = {1,0}
  if (fileData(1) < 2.69) {
    data(1) = {0,1};
  } else {
    data(1) = {1,0};
  }

  //Set the input configuration
  //----------------------------------------------------------------------------
  fileData = shift(fileData, -2);
  fileData.resize(fileData.n_elem - 2);
  data(0) = fileData;
}

void IsingDataLoader::loadData(uint32_t numberOfFiles, const char * fileNames[]) {
  set.set_size(numberOfFiles);
  for(uint32_t i = 0; i < numberOfFiles; ++i) {
    setData(set(i), fileNames[i]);
  }
}
