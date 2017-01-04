//Header for the ShallowNetwork class
#ifndef SHALLOWNETWORK_H
#define SHALLOWNETWORK_H
#include<armadillo>

class ShallowNetwork {

  //Class members
  //----------------------------------------------------------------------------
private:

  //Number of neurons in each layer
  uint32_t inputNeurons, hiddenNeurons, outputNeurons;

  //Neurons biases stored as vectors for each layer
  arma::fvec hiddenBias, outputBias, output;

  //weight matrices within layers
  arma::fmat weightInputHidden, weightHiddenOutput;

  //public Methods
  //----------------------------------------------------------------------------
public:

  //Constructor with default values for data members
  ShallowNetwork(uint32_t inputNeurons = 2500, uint32_t hiddenNeurons = 30, uint32_t outputNeurons = 2);

  //getters and setters
  arma::Col<uint32_t> getStructure() const;

  //As of now, I am not going to allow setters, initialize the network with the right value using the constructor.
  /*
  void setInputNeurons(uint32_t inputNeurons);
  void setHiddenNeurons(uint32_t hiddenNeurons);
  void setOutputNeurons(uint32_t outputNeurons);
  void setStructure(arma::Col<uint32_t> & structure);
*/

  //save to and load a network form the given directory. If used with no arguments it saves to and load from the same directory as the file.
  bool saveNetwork(const char * directoryName = ".");
  bool loadNetwork(const char * directoryName = ".");

  //get the result of the network evaluation;
  uint32_t getResult(arma::fvec & input);

  //to be inplemented later
  //----------------------------------------------------------------------------
  //training evaluation
  //double getAccuracy();
  //----------------------------------------------------------------------------

  //private methods
  //----------------------------------------------------------------------------
private:

  void initializeBiases();
  void initializeWeights();
  void activationFunction(arma::fvec & input);
  void feedForward(arma::fvec & input);
};

#endif
