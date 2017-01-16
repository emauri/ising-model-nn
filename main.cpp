/****************************************************************************************
Using LAPACK and OpenBLAS packages on Armadillo
Armadillo package version 7.600.2 installed via website: http://arma.sourceforge.net
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
#include <chrono>
#include <fstream>
#include <math.h>

#include "IsingDataLoader.h"
#include "NetworkTrainer.h"

using namespace arma;

#define SIZEF (sizeof(float))
#define TOTTEST 5000
#define TOTTRAIN 33000
#define TOTVAL 5000

#define INPUTNEURONS 2500
#define HIDDENNEURONS 100
#define OUTPUTNEURONS 2

#define LEARNINGRATE 0.01
#define NUMEPOCH 20
#define BATCHSIZE 100
#define USEVALIDATION true

unsigned int n_cores, n_queries;

void IsingParallelNN()
{
    bsp_begin(n_cores);

    /********************************************************************
     * INITIALIZATION:

     ********************************************************************/
    unsigned int pid = bsp_pid();

    // LOAD SEPERATE DATA SETS ON EACH CORE
    //----------------------------------------------------------------------------
    IsingDataLoader training;
    IsingDataLoader validation;
    IsingDataLoader test;

    test.loadData(TOTTEST, 0, 1, "dataList/testData.txt");
    training.loadData(TOTTRAIN, pid, n_cores, "dataList/trainingData.txt");
    validation.loadData(TOTVAL, 0, 1, "dataList/validationData.txt");

    // INITIALIZE NETWORK ON EACH CORE
    //----------------------------------------------------------------------------

    ShallowNetwork network(INPUTNEURONS, HIDDENNEURONS, OUTPUTNEURONS);

    // POINTER ARRAYS TO VEC AND MAT TYPES TO USE IN BSP FUNCTIONS
    float* localHiddenBias = network.hiddenBias.memptr();
    float* localOutputBias = network.outputBias.memptr();
    float* localWeightInputHidden = network.weightInputHidden.memptr();
    float* localWeightHiddenOutput = network.weightHiddenOutput.memptr();

    // REGISTER INTO STACK
    bsp_push_reg(localWeightInputHidden, HIDDENNEURONS * INPUTNEURONS * SIZEF);
    bsp_push_reg(localWeightHiddenOutput, OUTPUTNEURONS * HIDDENNEURONS * SIZEF);
    bsp_push_reg(localHiddenBias, HIDDENNEURONS * SIZEF);
    bsp_push_reg(localOutputBias, OUTPUTNEURONS * SIZEF);
    bsp_sync();

    // COMMUNICATE WEIGHTS AND BIASES FROM PROCESSOR 0 TO ALL OTHER PROCESSOR
    bsp_get(0, localWeightInputHidden, 0, localWeightInputHidden, HIDDENNEURONS * INPUTNEURONS * SIZEF);
    bsp_get(0, localWeightHiddenOutput, 0, localWeightHiddenOutput, OUTPUTNEURONS * HIDDENNEURONS * SIZEF);
    bsp_get(0, localHiddenBias, 0, localHiddenBias, HIDDENNEURONS * SIZEF);
    bsp_get(0, localOutputBias, 0, localOutputBias, OUTPUTNEURONS * SIZEF);
    bsp_sync();

    // SET WEIGHTS AND BIASES ON EACH CORE TO THE SAME VALUE AS ONES ON PROCESSOR 0

    if(pid == pid) {

        std::cout << "This is core " << pid << std::endl
                  << "Test data accuracy before training: " << network.getAccuracyOfSet(test.getDataSet()) << std::endl;
        bsp_sync();
    }

    double time0 = bsp_time();

    // auto t1 = std::chrono::high_resolution_clock::now();

    NetworkTrainer trainer(&network, LEARNINGRATE, 1, BATCHSIZE, USEVALIDATION);

    // TRAIN NETWORK ON EACH CORE
    //----------------------------------------------------------------------------
    for(uint32_t i = 0; i < NUMEPOCH; i++) {

        trainer.trainNetwork(i, NUMEPOCH, training.getDataSet(), validation.getDataSet());

        // COMMUNICATE WEIGHTS AND BIASES TO EVERY CORES AFTER EVERY EPOCH

        // VARIABLE TO STORE WEIGHT AND BIASES FROM OTHER CORES

        float* otherWeightInputHidden = new float[HIDDENNEURONS * INPUTNEURONS];
        float* otherWeightHiddenOutput = new float[OUTPUTNEURONS * HIDDENNEURONS];
        float* otherHiddenBias = new float[OUTPUTNEURONS * HIDDENNEURONS];
        float* otherOutputBias = new float[OUTPUTNEURONS * HIDDENNEURONS];

        arma::fmat CumWeightInputHidden(HIDDENNEURONS, INPUTNEURONS, fill::zeros);
        arma::fmat CumWeightHiddenOutput(OUTPUTNEURONS, HIDDENNEURONS, fill::zeros);
        arma::fvec CumHiddenBias(HIDDENNEURONS, fill::zeros);
        arma::fvec CumOutputBias(OUTPUTNEURONS, fill::zeros);

        // GET WEIGHT AND BIASES FROM EACH CORE, SAVE TO POINTER other(...)
        for(uint32_t j = 0; j < n_cores; j++) {

            bsp_get(j, localWeightInputHidden, 0, otherWeightInputHidden, HIDDENNEURONS * INPUTNEURONS * SIZEF);
            bsp_get(j, localWeightHiddenOutput, 0, otherWeightHiddenOutput, OUTPUTNEURONS * HIDDENNEURONS * SIZEF);
            bsp_get(j, localHiddenBias, 0, otherHiddenBias, HIDDENNEURONS * SIZEF);
            bsp_get(j, localOutputBias, 0, otherOutputBias, OUTPUTNEURONS * SIZEF);

            bsp_sync();

            // CUMMULATE WEIGHT AND BIAS
            CumWeightInputHidden += arma::fmat(otherWeightInputHidden, HIDDENNEURONS, INPUTNEURONS);
            CumWeightHiddenOutput += arma::fmat(otherWeightHiddenOutput, OUTPUTNEURONS, HIDDENNEURONS);
            CumHiddenBias += arma::fvec(otherHiddenBias, HIDDENNEURONS);
            CumOutputBias += arma::fvec(otherOutputBias, OUTPUTNEURONS);
        }

        // TAKE AVERAGE
        network.weightInputHidden = CumWeightInputHidden / n_cores;
        network.weightHiddenOutput = CumWeightHiddenOutput / n_cores;
        network.hiddenBias = CumHiddenBias / n_cores;
        network.outputBias = CumOutputBias / n_cores;
    }

    // TEST NETWORK ACCURACY ON DATASET
    //----------------------------------------------------------------------------
    bsp_sync();
    std::cout << "Training time: " << bsp_time() - time0 << "\n";

    // auto t2 = std::chrono::high_resolution_clock::now();

    if(pid == pid) {

        std::cout << "This is core " << pid << std::endl
                  << "Test data accuracy after training: " << network.getAccuracyOfSet(test.getDataSet()) << std::endl;
        //                  << "Training took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
        //                  t1).count()
        //                  << " milliseconds." << std::endl;
        bsp_sync();
    }
    network.saveNetwork(pid, ".");
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
    // n_cores = 1;

    std::cout << std::endl
              << " Neural network structure: " << std::endl
              << "==========================================================================" << std::endl
              << " LR: " << LEARNINGRATE << ", Number of Epochs: " << NUMEPOCH << ", Batch size: " << BATCHSIZE
              << std::endl
              << " " << INPUTNEURONS << " Input Neurons, " << HIDDENNEURONS << " Hidden Neurons, " << OUTPUTNEURONS
              << " Output Neurons" << std::endl
              << "==========================================================================" << std::endl
              << std::endl;

    // SEQUENTIAL VERSION, COMMENT OUT OTHER LINES CORRESPONDINGLY IF THIS VERSION IS USED
    //----------------------------------------------------------------------------
    //    IsingDataLoader training;
    //    IsingDataLoader validation;
    //    IsingDataLoader test;
    //
    //    test.loadData(TOTTEST, 0, 1, "dataList/testData.txt");
    //    training.loadData(TOTTRAIN, 0, 1, "dataList/trainingData.txt");
    //    validation.loadData(TOTVAL, 0, 1, "dataList/validationData.txt");
    //
    //    // initialize network and trainer
    //    //----------------------------------------------------------------------------
    //
    //    ShallowNetwork network(INPUTNEURONS, HIDDENNEURONS, OUTPUTNEURONS);
    //
    //    NetworkTrainer trainer(&network, LEARNINGRATE, NUMEPOCH, BATCHSIZE, USEVALIDATION);
    //
    //    // arma::field< arma::field<arma::fvec> > * p = training.getDataSet();
    //
    //    // train the network
    //    //----------------------------------------------------------------------------
    //    auto t1 = std::chrono::high_resolution_clock::now();
    //
    //    trainer.trainNetwork(0, NUMEPOCH, training.getDataSet(), validation.getDataSet());
    //
    //    auto t2 = std::chrono::high_resolution_clock::now();
    //
    //    std::cout << "Training took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
    //              << " milliseconds.\n";
    //
    //    // network.saveNetwork("data");
    //
    //    // test network accuracy
    //    //----------------------------------------------------------------------------
    //    std::cout << "Test data accuracy: " << network.getAccuracyOfSet(test.getDataSet()) << std::endl;

    IsingParallelNN();

    return 0;
}
