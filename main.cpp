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
#include <fstream>
#include <math.h>

#include "IsingDataLoader.h"
#include "NetworkTrainer.h"

using namespace arma;

#define SIZED (sizeof(double))
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
    validation.loadData(TOTVAL, pid, n_cores, "dataList/validationData.txt");

    // INITIALIZE NETWORK ON EACH CORE
    //----------------------------------------------------------------------------

    ShallowNetwork network(INPUTNEURONS, HIDDENNEURONS, OUTPUTNEURONS);

    // POINTER ARRAYS TO VEC AND MAT TYPES TO USE IN BSP FUNCTIONS
    double* localHiddenBias = network.hiddenBias.memptr();
    double* localOutputBias = network.outputBias.memptr();
    double* localWeightInputHidden = network.weightInputHidden.memptr();
    double* localWeightHiddenOutput = network.weightHiddenOutput.memptr();

    // REGISTER INTO STACK
    bsp_push_reg(localWeightInputHidden, HIDDENNEURONS * INPUTNEURONS * SIZED);
    bsp_push_reg(localWeightHiddenOutput, OUTPUTNEURONS * HIDDENNEURONS * SIZED);
    bsp_push_reg(localHiddenBias, HIDDENNEURONS * SIZED);
    bsp_push_reg(localOutputBias, OUTPUTNEURONS * SIZED);
    bsp_sync();

    // COMMUNICATE WEIGHTS AND BIASES FROM PROCESSOR 0 TO ALL OTHER PROCESSOR
    bsp_get(0, localWeightInputHidden, 0, localWeightInputHidden, HIDDENNEURONS * INPUTNEURONS * SIZED);
    bsp_get(0, localWeightHiddenOutput, 0, localWeightHiddenOutput, OUTPUTNEURONS * HIDDENNEURONS * SIZED);
    bsp_get(0, localHiddenBias, 0, localHiddenBias, HIDDENNEURONS * SIZED);
    bsp_get(0, localOutputBias, 0, localOutputBias, OUTPUTNEURONS * SIZED);
    bsp_sync();

    // CONVERT FROM ARRAY POINTER BACK TO VEC AND MAT TYPE
    arma::mat LocalWeightInputHidden(localWeightInputHidden, HIDDENNEURONS, INPUTNEURONS);
    arma::mat LocalWeightHiddenOutput(localWeightHiddenOutput, OUTPUTNEURONS, HIDDENNEURONS);
    arma::vec LocalHiddenBias(localHiddenBias, HIDDENNEURONS);
    arma::vec LocalOutputBias(localOutputBias, OUTPUTNEURONS);

    // SET WEIGHTS AND BIASES ON EACH CORE TO THE SAME VALUE AS ONES ON PROCESSOR 0
    network.weightInputHidden = LocalWeightInputHidden;
    network.weightHiddenOutput = LocalWeightHiddenOutput;
    network.hiddenBias = LocalHiddenBias;
    network.outputBias = LocalOutputBias;

    if(pid == pid) {

        std::cout << "This is core " << pid << std::endl
                  << "Test data accuracy before training: " << network.getAccuracyOfSet(test.getDataSet()) << std::endl;
        bsp_sync();
    }

    // TRAIN NETWORK ON EACH CORE
    //----------------------------------------------------------------------------
    for(uint32_t i = 0; i < NUMEPOCH; i++) {
        NetworkTrainer trainer(&network, LEARNINGRATE, 1, BATCHSIZE, USEVALIDATION);

        trainer.trainNetwork(i, NUMEPOCH, training.getDataSet(), validation.getDataSet());

        // COMMUNICATE WEIGHTS AND BIASES TO EVERY CORES AFTER EVERY EPOCH

        // VARIABLE TO STORE WEIGHT AND BIASES FROM OTHER CORES
        double* otherWeightInputHidden = new double[HIDDENNEURONS * INPUTNEURONS];
        double* otherWeightHiddenOutput = new double[OUTPUTNEURONS * HIDDENNEURONS];
        double* otherHiddenBias = new double[OUTPUTNEURONS * HIDDENNEURONS];
        double* otherOutputBias = new double[OUTPUTNEURONS * HIDDENNEURONS];

        // GET WEIGHT AND BIASES FROM EACH CORE, SAVE TO POINTER other(...)
        for(uint32_t j = 0; j < n_cores; j++) {
            if(pid != j) {
                bsp_get(j, localWeightInputHidden, 0, otherWeightInputHidden, HIDDENNEURONS * INPUTNEURONS * SIZED);
                bsp_get(j, localWeightHiddenOutput, 0, otherWeightHiddenOutput, OUTPUTNEURONS * HIDDENNEURONS * SIZED);
                bsp_get(j, localHiddenBias, 0, otherHiddenBias, HIDDENNEURONS * SIZED);
                bsp_get(j, localOutputBias, 0, otherOutputBias, OUTPUTNEURONS * SIZED);

                // CONVERT FROM ARRAY POINTER BACK TO VEC AND MAT TYPE
                arma::mat OtherWeightInputHidden(otherWeightInputHidden, HIDDENNEURONS, INPUTNEURONS);
                arma::mat OtherWeightHiddenOutput(otherWeightHiddenOutput, OUTPUTNEURONS, HIDDENNEURONS);
                arma::vec OtherHiddenBias(otherHiddenBias, HIDDENNEURONS);
                arma::vec OtherOutputBias(otherOutputBias, OUTPUTNEURONS);

                // SET WEIGHTS AND BIASES ON EACH CORE TO THE SAME VALUE AS ONES ON PROCESSOR 0
                network.weightInputHidden += OtherWeightInputHidden;
                network.weightHiddenOutput += OtherWeightHiddenOutput;
                network.hiddenBias += OtherHiddenBias;
                network.outputBias += OtherOutputBias;

                bsp_sync();
            }
        }

        // TAKE AVERAGE
        network.weightInputHidden = network.weightInputHidden / n_cores;
        network.weightHiddenOutput = network.weightHiddenOutput / n_cores;
        network.hiddenBias = network.hiddenBias / n_cores;
        network.outputBias = network.outputBias / n_cores;
    }

    // TEST NETWORK ACCURACY ON DATASET
    //----------------------------------------------------------------------------
    bsp_sync();

    if(pid == pid) {

        std::cout << "This is core " << pid << std::endl
                  << "Test data accuracy after training: " << network.getAccuracyOfSet(test.getDataSet()) << std::endl;
        bsp_sync();
    }

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
    //    // load data sets
    //    //----------------------------------------------------------------------------
    //    IsingDataLoader training;
    //    IsingDataLoader validation;
    //    IsingDataLoader test;
    //
    //    test.loadData(15,1,4, "dataList/testData.txt");
    //    training.loadData(16,3,4, "dataList/trainingData.txt");
    //    validation.loadData(16,2,4, "dataList/validationData.txt");
    //
    //    // initialize network and trainer
    //    //----------------------------------------------------------------------------
    //
    //    ShallowNetwork network(2500, 100, 2);
    //
    //    NetworkTrainer trainer(&network, 0.01, 5, 100, true);
    //  trainer.trainNetwork(training.getDataSet(), validation.getDataSet());
    ////
    ////    std::cout << "Test data accuracy: " << network.getAccuracyOfSet(test.getDataSet()) << std::endl;
    IsingParallelNN();

    return 0;
}
