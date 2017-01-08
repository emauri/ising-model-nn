//Implementation for the NetworkTrainer class
#include<iostream>
#include <math.h>
//Include header
#include "NetworkTrainer.h"

using namespace arma;

//constructor
NetworkTrainer::NetworkTrainer(ShallowNetwork * network, float lR, uint32_t nOE, uint32_t bS, bool uV) : network(network), learningRate(lR), numberOfEpochs(nOE), batchSize(bS), useValidation(uV) {

  //initialize deltaBias vectors
  deltaHiddenBias.zeros(network->inputNeurons);

  deltaOutputBias.zeros(network->outputNeurons);

  //initialize deltaWeight matrices
  deltaWeightInputHidden.zeros(network->hiddenNeurons, network->inputNeurons);

  deltaWeightHiddenOutput.zeros(network->outputNeurons, network->hiddenNeurons);

  //initialize vectors for monitoring progress
  trainingAccuracy.zeros(nOE);
  if (uV) {
    validationAccuracy.zeros(nOE);
    validationCost.zeros(nOE);
  }
}

//setters
void NetworkTrainer::setTrainingParameters(float learningRate, uint32_t numberOfEpochs, uint32_t batchSize, bool useValidation) {
  this->learningRate = learningRate;
  this->numberOfEpochs = numberOfEpochs;
  this->batchSize = batchSize;
  this->useValidation = useValidation;
}

//conpute output error using cross-entropy cost function
fvec NetworkTrainer::getOutputError(fvec & output, fvec & label) {

  //return output error
  return (output - label);
}

//implementation of backpropagation algorithm
void NetworkTrainer::backpropagation(fvec & input, fvec & label) {

  //feed forward
  network->feedForward(input);

  //compute error in the output
  fvec delta = getOutputError(network->output, label);

  //add error in the output biases
  deltaOutputBias += delta;

  //compute error in the hidden-output weights
  deltaWeightHiddenOutput += dot(delta, network->hidden.t());

  //compute error of the hidden layers
  delta = dot(network->weightHiddenOutput.t(), delta) * (network->hidden) * (1 - network->hidden); //using sigmoid activation function

  //add error in the hidden biases
  deltaHiddenBias += delta;

  //compute error in the input-hidden weights
  deltaWeightInputHidden += dot(delta, input.t());
}

//update network weights and biases
void NetworkTrainer::updateNetwork(field< field<fvec> > * trainingSet, uint32_t currentBatchStart, uint32_t size) {

  //if the current batch is not the first one, reset all the errors in weights and biases to zero
  if (currentBatchStart) {
    deltaHiddenBias.zeros();
    deltaOutputBias.zeros();
    deltaWeightInputHidden.zeros();
    deltaWeightHiddenOutput.zeros();
  }

  uint32_t stop = (currentBatchStart + batchSize > size) ? size : currentBatchStart + batchSize;

  for (uint32_t i = currentBatchStart; i < stop; ++i) {

    //backpropagation
    backpropagation(trainingSet->at(shuffleData(i))(0), trainingSet->at(shuffleData(i))(1));

    //check output against label
    if (network->getResult(network->output) != trainingSet->at(shuffleData(i))(1).index_max()) {
      ++incorrectResults;
    }
  }

  //update weights
  float prefactor = learningRate / batchSize;

  network->weightInputHidden -= prefactor * deltaWeightInputHidden;

  network->weightHiddenOutput -= prefactor * deltaWeightHiddenOutput;

  //update biases
  network->hiddenBias -= prefactor * deltaHiddenBias;

  network->outputBias -= prefactor * deltaOutputBias;
}

//implement stocastic gradient descent to train the network
void NetworkTrainer::stochasticGradientDescent(field< field<fvec> > * trainingSet, uint32_t size) {

  //shuffle data in the training set
  shuffleData = shuffle(shuffleData);

  incorrectResults = 0;

  //update network based on gradient descent on a mininbatch

  //this 'for loop' constitute an epoch
  for (uint32_t i = 0; i < size; i += batchSize) {
    updateNetwork(trainingSet, i, size);
  }
}

//train the neural network
void NetworkTrainer::trainNetwork(field< field <fvec> > * trainingSet, field < field<fvec> > * validationSet) {

  //initialize shuffleData
  uint32_t trainingSize = trainingSet->n_elem;
  shuffleData.set_size(trainingSize);
  for (uint32_t i = 0; i < trainingSize; ++i) {
    shuffleData(i) = i;
  }
  //loop over the number of epochs
  for (uint32_t i = 0; i < numberOfEpochs; ++i) {
    stochasticGradientDescent(trainingSet, trainingSize);

    //store training accuracy for the epoch
    trainingAccuracy(i) = 100 - (incorrectResults/trainingSize * 100);

    if (useValidation && validationSet != NULL) {
      validationAccuracy(i) = network->getAccuracyOfSet(validationSet);
      validationCost(i) = monitorCost(validationSet);
    }

    //print accuracy for the epoch
    std::cout << "==========================================================================" << std::endl
    << "Epoch: " << i << " of " << numberOfEpochs << std::endl
    << " Training set accuracy: " << trainingAccuracy(i) << std::endl;
    if (useValidation && validationSet != NULL) {
      std::cout << "Validation set accuracy: " << validationAccuracy(i) << " Total cost: " << validationCost(i) << std::endl;
    }
    std::cout << "==========================================================================" << std::endl;

  }
}

//define cross-entropy cost function
float NetworkTrainer::crossEntropy(fvec & output, fvec & label) {

  //define vector of ones with size as output
  fvec ones;
  ones.ones(output.n_elem);

  return accu(-label * log(output) + (label - ones) * log(ones - output));
}

float NetworkTrainer::monitorCost(field< field<fvec> > * set) {

  float totalCost = 0;

  uint32_t size = set->n_elem;

  for (uint32_t i = 0; i < size; ++i) {
    network->feedForward(set->at(i)(0)); //feedforward to compute output
    totalCost += crossEntropy(network->output, set->at(i)(1));
  }
  return totalCost;
}
