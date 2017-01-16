// Header for the ShallowNetwork class
#ifndef SHALLOWNETWORK_H
#define SHALLOWNETWORK_H
#include <armadillo>

// friend class
class NetworkTrainer;

class ShallowNetwork
{

    // Class members
    //----------------------------------------------------------------------------
public:
    // Number of neurons in each layer
    uint32_t inputNeurons, hiddenNeurons, outputNeurons;

    // Neurons layers as vectors
    arma::fvec hidden, output;

    // Neurons biases stored as vectors for each layer
    arma::fvec hiddenBias, outputBias;

    // Weight matrices within layers
    arma::fmat weightInputHidden, weightHiddenOutput;

    // Public Methods
    //----------------------------------------------------------------------------
public:
    // Constructor with default values for data members
    ShallowNetwork(uint32_t inputNeurons = 2500, uint32_t hiddenNeurons = 100, uint32_t outputNeurons = 2);

    // getters and setters
    arma::Col<uint32_t> getStructure() const;

    // As of now, I am not going to allow setters, initialize the network with the right values using the constructor.
    /*
    void setInputNeurons(uint32_t inputNeurons);
    void setHiddenNeurons(uint32_t hiddenNeurons);
    void setOutputNeurons(uint32_t outputNeurons);
    void setStructure(arma::Col<uint32_t> & structure);
  */

    // save to and load a network form the given directory. If used with no arguments it saves to and load from the same
    // directory as the file.
    bool saveNetwork(int pid, const char* directoryName = ".");
    bool loadNetwork(const char* directoryName = ".");

    // get the result of the network evaluation: feedforward input -> return the index of the neuron with the highest
    // output value;
    uint32_t getResult(arma::fvec& input);

    // training evaluation
    float getAccuracyOfSet(arma::field<arma::field<arma::fvec> >* set);

    // Friends
    //--------------------------------------------------------------------------------------------
    friend NetworkTrainer;

    // Private methods
    //----------------------------------------------------------------------------
private:
    void initializeBiases();
    void initializeWeights();
    void activationFunction(arma::fvec& input);
    void feedForward(arma::fvec& input);
};

#endif
