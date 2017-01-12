OBJS = main.o ShallowNetwork.o IsingDataLoader.o NetworkTrainer.o
CC = g++ --std=c++11
OPTIMIZE = -O3
CFLAGS = -Wall -c $(OPTIMIZE)
LFLAGS = -Wall $(OPTIMIZE)
LIBS = -Iarmadillo-7.600.2/include -I/opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas -llapack
#LIBS = -Iarmadillo-7.600.2/include -framework Accelerate
LINKOB = -Iarmadillo-7.600.2/include -I/opt/OpenBLAS/include/

main : $(OBJS)
		$(CC) $(LFLAGS) $(OBJS) -o main $(LIBS)

main.o : main.cpp ShallowNetwork.h IsingDataLoader.h NetworkTrainer.h
		$(CC) $(CFLAGS) main.cpp $(LINKOB)

ShallowNetwork.o : ShallowNetwork.h ShallowNetwork.cpp NetworkTrainer.h
		$(CC) $(CFLAGS) ShallowNetwork.cpp $(LINKOB)

IsingDataLoader.o : IsingDataLoader.h IsingDataLoader.cpp
		$(CC) $(CFLAGS) IsingDataLoader.cpp $(LINKOB)

NetworkTrainer.o : NetworkTrainer.h NetworkTrainer.cpp ShallowNetwork.h
		$(CC) $(CFLAGS) NetworkTrainer.cpp $(LINKOB)

clean:
		\rm *.o main
