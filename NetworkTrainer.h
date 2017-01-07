//Header for the NetworkTrainer class
#ifndef NETWORKTRAINER_H
#define NETWORKTRAINER_H

#include<armadillo>

#include "ShallowNetwork.h"

class NetworkTrainer {

  //Class members
  //----------------------------------------------------------------------------
private:
  //Network to train
  ShallowNetwork * network;

  //vectors storing changes in biases during gradient descent
  arma::fvec deltaHiddenBias, deltaOutputBias;

  //matrices storing cahnges in weights during gradient descent
  arma::fmat deltaWeightInputHidden, deltaWeightHiddenOutput;

  //learning parameters
  float learningRate;
  uint32_t numberOfEpochs;
  uint32_t batchSize;
  bool monitorTrainingAccuracy;
  bool monitorValidationAccuracy;

  //public Methods
  //----------------------------------------------------------------------------
public:

  //constructor with default values
  NetworkTrainer(ShallowNetwork * network, float learningRate = 0.01; uint32_t numberOfEpochs = 30, uint32_t miniBatchSize = 10, bool monitorTrainingAccuracy = false, bool monitorValidationAccuracy = false);

  //setters
  void setTrainingParameters(float learningRate, uint32_t numberOfEpochs, uint32_t batchSize);

  //private Methods
  //----------------------------------------------------------------------------
private:
  void stochasticGradientDescent();
  void updateNetwork();
  void backpropagation();
  void readData();

};

#endif
