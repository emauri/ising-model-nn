#include <iostream>

#include "NetworkTrainer.h"
#include "IsingDataLoader.h"

int main(int argc, const char * argv[]) {

  //load data sets
  //----------------------------------------------------------------------------
  IsingDataLoader training;
  IsingDataLoader validation;
  IsingDataLoader test;

  test.loadData(4970, "dataList/testData.txt");
  training.loadData(33000, "dataList/trainingData.txt");
  validation.loadData(5000, "dataList/validationData.txt");

  //initialize network and trainer
  //----------------------------------------------------------------------------

  ShallowNetwork network(2500, 100, 2);


  NetworkTrainer trainer(&network, 0.01, 20, 100, true);

  //arma::field< arma::field<arma::fvec> > * p = training.getDataSet();

  //train the network
  //----------------------------------------------------------------------------
  trainer.trainNetwork( training.getDataSet(), validation.getDataSet());

  //network.saveNetwork("data");

  //test network accuracy
  //----------------------------------------------------------------------------
  std::cout << "Test data accuracy: " << network.getAccuracyOfSet( test.getDataSet() ) << std::endl;

  return 0;
}
