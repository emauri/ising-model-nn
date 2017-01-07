//Implementation for the NetworkTrainer class
#include<iostream>
#include <math.h>
//Include header
#include "NetworkTrainer.h"

using namespace arma;

//constructor
NetworkTrainer::NetworkTrainer(ShallowNetwork * network, float lR; uint32_t nOE, uint32_t bS, bool monitorTA, bool monitorVA) : network(network), learningRate(lR), numberOfEpochs(nOE), batchSize(bS), monitorTrainingAccuracy(monitorTA), monitorValidationAccuracy(monitorVA) {

  //initialize deltaBias vectors
  deltaHiddenBias.zeros(network->inputNeurons);

  deltaOutputBias.zeros(network->outputNeurons);

  //initialize deltaWeight matrices
  deltaWeightInputHidden.zeros(network->hiddenNeurons, network->inputNeurons);

  deltaWeightHiddenOutput.zeros(network->outputNeurons, network->hiddenNeurons);
}

//setters
void NetworkTrainer::setTrainingParameters(learningRate, numberOfEpochs, batchSize) {
  this->learningRate = learningRate;
  this->numberOfEpochs = numberOfEpochs;
  this->batchSize = batchSize;
}

//conpute output error using cross-entropy cost function
fvec NetworkTrainer::getOutputError(fvec & output; fvec & label) {

  //return output error
  return (output - label);
}

//implementation of backpropagation algorithm
void NetworkTrainer::backpropagation() {

  //feed forward
  network->feedForward(network->input);

  //compute error in the output
  fvec delta = getOutputError(networ->ouput, label);

  //add error in the output biases
  deltaOutputBias += delta;

  //compute error in the hidden-output weights
  deltaWeightHiddenOutput += dot(delta, network->hidden.t());

  //compute error of the hidden layers
  delta = dot(network->weightHiddenOutput.t(), delta) * (network->hidden) * (1 - network->hidden); //using sigmoid activation function

  //add error in the hidden biases
  deltaHiddenBias += delta;

  //compute error in the input-hidden weights
  deltaWeightInputHidden += dot(delta, network->input.t());
}

//update network weights and biases (online learning)
void NetworkTrainer::updateNetwork(uint32_t currentBatchStart) {

  //if the current batch is not the first one, reset all the errors in weights and biases to zero
  if (currentBatchStart) {
    deltaHiddenBias.zeros();
    deltaOutputBias.zeros();
    deltaWeightInputHidden.zeros();
    deltaWeightHiddenOutput.zeros();
  }

  for (uint32_t i = 0; i < batchSize; ++i) {
    backpropagation();
  }

  //update weights
  float prefactor = learningRate / batchSize;

  network->weightInputHidden -= prefactor * deltaWeightInputHidden;

  network->weightHiddenOutput -= prefactor * deltaWeightHiddenOutput;

  //update biases
  network->hiddenBias -= prefactor * deltaHiddenBias;

  networl->outputBias -= prefactor * deltaOutputBias;
}

//implement stocastic gradient descent to train the network
void NetworkTrainer::stochasticGradientDescent(field<fvec> & trainingSet) {

  //shuffle data in the training set
  trainingSet = shuffle(trainingSet);

  //update network based on gradient descent on a mininbatch
  uint32_t size = trainingSet.n_elem;
  for (uint32_t i = 0; i < size; ++batchSize) {
    updateNetwork(i);
  }
}
