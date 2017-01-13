//Implementation for the ShallowNetwork class: We have ONE hidden layer
#include<iostream>
#include <fstream>
#include <math.h>
//Include header
#include "ShallowNetwork.h"

using namespace arma;

//constructor
ShallowNetwork::ShallowNetwork(uint32_t iN, uint32_t hN, uint32_t oN) : inputNeurons(iN), hiddenNeurons(hN), outputNeurons(oN) {

  //initialize neurons to zero
  hidden.zeros(hN);
  output.zeros(oN);

  //initialize biases vectors
  initializeBiases();
  //initialize weights matrices
  initializeWeights();
}

//getters for the Network structure. It returns the network topology as a column vector.
Col<uint32_t> ShallowNetwork::getStructure() const {
  Col<uint32_t> structure = {inputNeurons, hiddenNeurons, outputNeurons};
  return structure;
}

/*
//setters
void ShallowNetwork::setInputNeurons(uint32_t inputNeurons) {
  this->inputNeurons = inputNeurons;
  initializeWeights();
  initializeBiases();
}

void ShallowNetwork::setHiddenNeurons(uint32_t hiddenNeurons) {
  this->hiddenNeurons = hiddenNeurons;
  initializeWeights();
  initializeBiases();
}

void ShallowNetwork::setOutputNeurons(uint32_t hiddenNeurons) {
  this->outputNeurons = outputNeurons;
  initializeWeights();
  initializeBiases();
}

void ShallowNetwork::setStructure(Col<uint32_t> & structure) {
  if (size(structure) != 3) {
    std::cout << "Wrong vector length. The vector should have length 3." << std::endl;
    return;
  }
  inputNeurons = structure(0);
  hiddenNeurons = structure(1);
  outputNeurons = structure(2);
  initializeWeights();
  initializeBiases();
}
*/

//biases initializer
void ShallowNetwork::initializeBiases() {

  //CAUTION, random value must scale as sqrt(N)
  //set the seed to a random value
  arma_rng::set_seed_random();

  //Biases initialization using Normal distribution
  hiddenBias.randn(hiddenNeurons);
  outputBias.randn(outputNeurons);
}

//Weights initializer
void ShallowNetwork::initializeWeights() {

  //set the seed to a random value
  arma_rng::set_seed_random();

  //set weights between input and hidden layers
  //----------------------------------------------------------------------------

  //CAUTION, random value must scale as sqrt(N)
  //standard deviation of the Gaussian distribution
  float stdDev = 1.0 / (float)sqrt(inputNeurons);

  //initialize the weights and rescale them
  weightInputHidden.randn(hiddenNeurons, inputNeurons);
  weightInputHidden *= stdDev;

  //set weights between hidden and output layers
  //----------------------------------------------------------------------------
  //CAUTION, random value must scale as sqrt(N)
  //standard deviation of the Gaussian
  stdDev = 1 / (float) sqrt(hiddenNeurons);

  //initialize the weights and rescale them
  weightHiddenOutput.randn(outputNeurons, hiddenNeurons);
  weightHiddenOutput *= stdDev;
}

//Save network configuration
bool ShallowNetwork::saveNetwork(const char * directoryName) {

  //convert directoryName to std::string
  std::string stringName = std::string(directoryName);

  //save weights matrices to .txt file in arma_binary format
  //----------------------------------------------------------------------------

  //save input-hidden weigths matrix
  bool ihWStatus = weightInputHidden.save(stringName + "/ih_weights.txt");

  //save hidden-output weights matrix
  bool hoWStatus = weightHiddenOutput.save(stringName + "/ho_weights.txt");

  //save biases vectors to .txt file in arma_binary format
  //----------------------------------------------------------------------------

  //save hidden layer biases vector
  bool hBStatus = hiddenBias.save(stringName + "/h_bias.txt");

  //save output layer biases vector
  bool oBStatus = outputBias.save(stringName + "/o_bias.txt");

  return (hoWStatus && ihWStatus && hBStatus && oBStatus);
}

//load network configuration
bool ShallowNetwork::loadNetwork(const char * directoryName) {

  //convert directoryName to std::string
  std::string stringName = std::string(directoryName);

  //load weights matrices from .txt file in arma_binary format
  //----------------------------------------------------------------------------

  //load input-hidden weigths matrix
  bool ihWStatus = weightInputHidden.load(stringName + "/ih_weights.txt", arma_binary);

  //load hidden-output weigths matrix
  bool hoWStatus = weightHiddenOutput.load(stringName + "/ho_weights.txt", arma_binary);

  //save biases vectors to .txt file in arma_binary format
  //----------------------------------------------------------------------------

  //load hidden layer biases vector
  bool hBStatus = hiddenBias.load(stringName + "/h_bias.txt", arma_binary);

  //load output layer biases vector
  bool oBStatus = outputBias.load(stringName + "/o_bias.txt", arma_binary);

  return (hoWStatus && ihWStatus && hBStatus && oBStatus);
}

//Activation function
void ShallowNetwork::activationFunction(fvec & input) {
  
  //may use TANH function or linear?
  //sigmoid function
  input = 1.0 / (1.0 + exp(-input));
}

//Feed Forward procedure
void ShallowNetwork::feedForward(fvec & input) {

  //calculate output from hidden layer
  //----------------------------------------------------------------------------
  hidden = weightInputHidden * input + hiddenBias;
  activationFunction(hidden);

  //calculate output
  //----------------------------------------------------------------------------
  output = weightHiddenOutput * hidden + outputBias;
  activationFunction(output);
}

//get the output neuron with the highest output value
uint32_t ShallowNetwork::getResult(fvec & input) {

  //feedforward input
  ShallowNetwork::feedForward(input);

  //return the index of the neuron with the highest output value
  return index_max(output);
}

//accuracy on an input set
float ShallowNetwork::getAccuracyOfSet(field< field<fvec> > * set) {
  float incorrectResults = 0;

  //compare result of each input with corresponding label
  uint32_t size = set->n_elem;
  for (uint32_t i = 0; i < set->n_elem; ++i) {
    if (getResult(set->at(i)(0)) != set->at(i)(1).index_max()) {
       ++incorrectResults;
    }
  }

  //return percentage of correct results
  return 100 - (incorrectResults / size * 100);
}
