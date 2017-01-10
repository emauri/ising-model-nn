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
void IsingDataLoader::setData(field<fvec> & data, std::string & filename) {

  //set data size to 2 (input configuration, output label)
  data.set_size(2);

  //load the data in the file, the first line contain the size of the Ising lattice,
  //the second line contains the temperature of the system. From the third line
  //onwards the file contains the Ising spin configuration
  fvec fileData;
  fileData.load(filename);

  //Set the output label
  //----------------------------------------------------------------------------
  //If the temperature is less than 2.269 it is below the critical temperature: ouput = {0,1},
  //otherwise it is above: output = {1,0}
  if (fileData(1) < 2.269) {
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

bool IsingDataLoader::loadData(uint32_t numberOfFiles, const char * listFile) {
  set.set_size(numberOfFiles);

  //check if all files are loaded
  bool allLoaded = true;

  //read data files names from file
  std::ifstream myfile (listFile);
  std::string filename;
  if (myfile.is_open()) {
    for(uint32_t i = 0; i < numberOfFiles; ++i) {
      if (std::getline(myfile, filename)) {
        setData(set(i), filename);
      }
      else {
        allLoaded = false;
      }
    }
    if (std::getline(myfile, filename)) {
      std::cout << "Warning: loaded less configurations than available." << std::endl;
      allLoaded = false;
    }
    myfile.close();
  }
  else {
    std::cout << "Couldn't open the file." << std::endl;
    allLoaded = false;
  }

  if (allLoaded) {
    std::cout << "Set properly loaded." << std::endl;
  } else {
    std::cout << "Unable to properly load the data set." << std::endl;
  }
  return allLoaded;
}

bool IsingDataLoader::saveDataSet(const char * fileName) {
  return set.save(fileName);
}

//data loader for previously saved set
bool IsingDataLoader::loadData(const char * fileName) {
  bool loaded = set.load(fileName);
  if (loaded) {
    std::cout << "Set properly loaded." << std::endl;
  }
  else {
    std::cout << "Unable to load the data set." << std::endl;
  }
  return loaded;
}
