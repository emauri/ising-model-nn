// Header for the NetworkTrainer class
#ifndef NETWORKTRAINER_H
#define NETWORKTRAINER_H

#include <armadillo>

#include "ShallowNetwork.h"

class NetworkTrainer
{

    // Class members
    //----------------------------------------------------------------------------
private:
    // Network to train
    ShallowNetwork* network;

    // vectors storing changes in biases during gradient descent
    arma::vec deltaHiddenBias, deltaOutputBias;

    // matrices storing cahnges in weights during gradient descent
    arma::mat deltaWeightInputHidden, deltaWeightHiddenOutput;

    // learning parameters
    float learningRate;
    uint32_t numberOfEpochs;
    uint32_t batchSize;
    // bool monitorTrainingCost;
    bool useValidation;

    // monitor progress during learning
    float incorrectResults = 0;

    arma::vec trainingAccuracy;
    arma::vec validationAccuracy;
    arma::vec validationCost;

    // store shuffled indeces to access data in random order
    arma::Col<uint32_t> shuffleData;

    // public Methods
    //----------------------------------------------------------------------------
public:
    // constructor with default values
    NetworkTrainer(ShallowNetwork* network,
        float learningRate = 0.01,
        uint32_t numberOfEpochs = 30,
        uint32_t miniBatchSize = 10,
        bool useValidation = false);

    // setters
    void
    setTrainingParameters(float learningRate, uint32_t numberOfEpochs, uint32_t batchSize, bool useValidation = false);

    // getters for monitorig vector
    arma::vec getTrainingAccuracy() const;
    arma::vec getValidationAccuracy() const;
    arma::vec getValidationCost() const;

    // network trainer
    void trainNetwork(uint32_t currentEpoch,
        uint32_t totalEpochs,
        arma::field<arma::field<arma::vec> >* trainingSet,
        arma::field<arma::field<arma::vec> >* validationSet = NULL);

    // private Methods
    //----------------------------------------------------------------------------
private:
    arma::vec getOutputError(arma::vec& output, arma::vec& label);
    void stochasticGradientDescent(arma::field<arma::field<arma::vec> >* trainingSet, uint32_t size);
    void updateNetwork(arma::field<arma::field<arma::vec> >* trainingSet, uint32_t currentBatchStart, uint32_t size);
    void backpropagation(arma::vec& input, arma::vec& label);
    float crossEntropy(arma::vec& output, arma::vec& label);
    float monitorCost(arma::field<arma::field<arma::vec> >* set);
};

#endif
