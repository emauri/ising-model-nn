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

    // Load seperate data sets on each core
    //----------------------------------------------------------------------------
    IsingDataLoader training;
    IsingDataLoader validation;
    IsingDataLoader test;

    test.loadData(70, "dataList/testData.txt");
    training.loadData(330, "dataList/trainingData.txt");
    validation.loadData(50, "dataList/validationData.txt");

    // Initialize network and trainer on each core
    //----------------------------------------------------------------------------

    ShallowNetwork network(2500, 100, 2);

    NetworkTrainer trainer(&network, 0.01, 5, 100, true);

    // train the network on each core
    //----------------------------------------------------------------------------
    trainer.trainNetwork(training.getDataSet(), validation.getDataSet());

    // test network accuracy on dataset
    //----------------------------------------------------------------------------
    bsp_sync();

    if(pid == pid) {
        std::cout << "This is core " << pid << std::endl
                  << "Test data accuracy: " << network.getAccuracyOfSet(test.getDataSet()) << std::endl;
    }

    // Send weights and biases to 
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
    //    // load data sets
    //    //----------------------------------------------------------------------------
    //    IsingDataLoader training;
    //    IsingDataLoader validation;
    //    IsingDataLoader test;
    //
    //    test.loadData(50, "dataList/testData.txt");
    //    training.loadData(330, "dataList/trainingData.txt");
    //    validation.loadData(50, "dataList/validationData.txt");
    //
    //    // initialize network and trainer
    //    //----------------------------------------------------------------------------
    //
    //    ShallowNetwork network(2500, 100, 2);
    //
    //    NetworkTrainer trainer(&network, 0.01, 5, 100, true);
    //    trainer.trainNetwork(training.getDataSet(), validation.getDataSet());
    //
    //    std::cout << "Test data accuracy: " << network.getAccuracyOfSet(test.getDataSet()) << std::endl;
    IsingParallelNN();

    return 0;
}
