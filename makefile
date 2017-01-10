OBJS = main.o ShallowNetwork.o IsingDataLoader.o NetworkTrainer.o
CC = g++ --std=c++11
OPTIMIZE = -O3
CFLAGS = -Wall -c $(OPTIMIZE)
LFLAGS = -Wall $(OPTIMIZE)
LIBS = -larmadillo -I/opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas -llapack
main : $(OBJS)
		$(CC) $(LFLAGS) $(OBJS) -o main $(LIBS)

main.o : main.cpp ShallowNetwork.h IsingDataLoader.h NetworkTrainer.h
		$(CC) $(CFLAGS) main.cpp

ShallowNetwork.o : ShallowNetwork.h ShallowNetwork.cpp NetworkTrainer.h
		$(CC) $(CFLAGS) ShallowNetwork.cpp

IsingDataLoader.o : IsingDataLoader.h IsingDataLoader.cpp
		$(CC) $(CFLAGS) IsingDataLoader.cpp

NetworkTrainer.o : NetworkTrainer.h NetworkTrainer.cpp ShallowNetwork.h
		$(CC) $(CFLAGS) NetworkTrainer.cpp

clean:
		\rm *.o main
