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

  //vec tors storing changes in biases during gradient descent
  arma::vec deltaHiddenBias, deltaOutputBias;

  //matrices storing changes in weights during gradient descent
  arma::mat deltaWeightInputHidden, deltaWeightHiddenOutput;

  //learning parameters
  double learningRate;
  uint32_t numberOfEpochs;
  uint32_t batchSize;
  double regularizer;

  //bool monitorTrainingCost;
  bool useValidation;

  //monitor progress during learning
  double incorrectResults = 0;

  arma::vec trainingAccuracy;
  arma::vec validationAccuracy;
  arma::vec validationCost;

  //store shuffled indeces to access data in random order
  arma::Col<uint32_t> shuffleData;

  //public Methods
  //----------------------------------------------------------------------------
public:

  //constructor with default values
  NetworkTrainer(ShallowNetwork * network, double learningRate = 0.01, uint32_t numberOfEpochs = 30, uint32_t miniBatchSize = 10, double regularizer = 0.0, bool useValidation = false);

  //setters
  void setTrainingParameters(double learningRate, uint32_t numberOfEpochs, uint32_t batchSize, double regularizer = 0.0, bool useValidation = false);

  //getters for monitorig vec tor
  arma::vec getTrainingAccuracy() const;
  arma::vec getValidationAccuracy() const;
  arma::vec getValidationCost() const;

  //network trainer
  void trainNetwork(arma::field< arma::field<arma::vec > > * trainingSet, arma::field< arma::field<arma::vec > > * validationSet = NULL);

  //private Methods
  //----------------------------------------------------------------------------
private:

  arma::vec getOutputError(arma::vec & output, arma::vec & label);
  void stochasticGradientDescent(arma::field< arma::field<arma::vec > > * trainingSet, uint32_t size);
  void updateNetwork(arma::field< arma::field<arma::vec > > * trainingSet, uint32_t currentBatchStart, uint32_t size);
  void backpropagation(arma::vec & input, arma::vec & label);
  double crossEntropy(arma::vec & output, arma::vec & label);
  double monitorCost(arma::field< arma::field<arma::vec > > * set);
};

#endif
