/****************************************************************************************
Using LAPACK and OpenBLAS packages on Armadillo
Armadillo package version 7.600.2 installed via website: http://arma.sourceforge.net/download.html
LAPACK-dev and OpenBLAS-dev package installed via apt-get

Inside file ¨config.hpp¨ in /usr/local/include/armadillo_bits:
1. Uncomment line #define ARMA_USE_LAPACK
2. Comment out line #define ARMA_USE_WRAPPER

Compiler option: -std=c++11;-O3; -llapack; -lopenblas
Linker option: -pthread -lrt -lm
Library path: /home/robert/MulticoreBSP-for-C/lib/
Libraries: libmcbsp1.2.0.a
*****************************************************************************************/

#include "mcbsp.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdio.h>

#include <armadillo>
#include <fstream>
#include <math.h>

#include "IsingDataLoader.h"
#include "NetworkTrainer.h"

using namespace arma;

#define SIZEUI (sizeof(unsigned int))

unsigned int n_cores;

void IsingParallelNN()
{
    bsp_begin(n_cores);

    /********************************************************************
     * INITIALIZATION:

     ********************************************************************/
    unsigned int pid = bsp_pid();

    // Number of neurons in each layer
    uint32_t inputNeurons, hiddenNeurons, outputNeurons;
    // Neurons layers as vectors
    arma::fvec hidden, output;
    // Neurons biases stored as vectors for each layer
    arma::fvec hiddenBias, outputBias;
    // Weight matrices within layers
    arma::fmat weightInputHidden, weightHiddenOutput;

    // vectors storing changes in biases during gradient descent
    arma::fvec deltaHiddenBias, deltaOutputBias;
    // matrices storing cahnges in weights during gradient descent
    arma::fmat deltaWeightInputHidden, deltaWeightHiddenOutput;
    // learning parameters
    float learningRate;
    uint32_t numberOfEpochs;
    uint32_t batchSize;
    // bool monitorTrainingCost;
    bool useValidation;
    // monitor progress during learning
    float incorrectResults = 0;

    arma::fvec trainingAccuracy;
    arma::fvec validationAccuracy;
    arma::fvec validationCost;

    // store shuffled indeces to access data in random order
    arma::Col<uint32_t> shuffleData;

    // Register into stack
    /*    bsp_push_reg(query_rep, n_queries * SIZEUI);
        bsp_push_reg(Query_rep, n_queries * n_cores * SIZEUI);
        bsp_sync();*/

    bsp_sync();
    bsp_end();
}


/********************************************************************
 * MAIN Function:

 ********************************************************************/

int main(int argc, char* argv[])
{
    bsp_init(IsingParallelNN, argc, argv);

    n_cores = bsp_nprocs();
    // load data sets
    //----------------------------------------------------------------------------
    IsingDataLoader training;
    IsingDataLoader validation;
    IsingDataLoader test;

    test.loadData(4970, "dataList/testData.txt");
    training.loadData(33000, "dataList/trainingData.txt");
    validation.loadData(5000, "dataList/validationData.txt");

    // initialize network and trainer
    //----------------------------------------------------------------------------

    ShallowNetwork network(2500, 100, 2);

    NetworkTrainer trainer(&network, 0.01, 20, 100, true);
    IsingParallelNN();

    return 0;
}
