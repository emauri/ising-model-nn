#include <iostream>
#include<chrono>

#include "NetworkTrainer.h"
#include "IsingDataLoader.h"
//#include "cblas.h"

int main(int argc, const char * argv[]) {

  //openblas_set_num_threads(4);

  //load data sets
  //----------------------------------------------------------------------------
  IsingDataLoader training;
  IsingDataLoader validation;
  IsingDataLoader test;

  if ( !test.loadData(5000, "dataList/testData.txt") ) { return -1; };
  if( !training.loadData(33000, "dataList/trainingData.txt") ) { return -1; }
  if( !validation.loadData(5000, "dataList/validationData.txt") ) { return -1; };

  //initialize network and trainer
  //----------------------------------------------------------------------------
  ShallowNetwork network(2500, 100, 2);

  NetworkTrainer trainer(&network, 0.01, 20, 33000, 0.5, true);

  //Test data accuracy before training
  //----------------------------------------------------------------------------
  std::cout << std::endl;
  std::cout << "Test data accuracy: " << network.getAccuracyOfSet( test.getDataSet() ) << std::endl;

  //arma::field< arma::field<arma::vec> > * p = training.getDataSet();

  //train the network
  //----------------------------------------------------------------------------
  auto t1 = std::chrono::high_resolution_clock::now();

    trainer.trainNetwork( training.getDataSet(), validation.getDataSet());

  auto t2 = std::chrono::high_resolution_clock::now();
   std::cout << "Training took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " milliseconds.\n";


  network.saveNetwork("data");

  //test network accuracy
  //----------------------------------------------------------------------------
  std::cout << "Test data accuracy: " << network.getAccuracyOfSet( test.getDataSet() ) << std::endl;

  return 0;
}
