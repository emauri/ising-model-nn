OBJS = main.o ShallowNetwork.o IsingDataLoader.o NetworkTrainer.o
CC = g++ --std=c++11
OPTIMIZE = -O3
CFLAGS = -Wall -c $(OPTIMIZE)
LFLAGS = -Wall $(OPTIMIZE)

#Use LIBS with -framework Accelerate on Mac OsX, on Linux with OpenBlas uncomment the links to
# openBlas (on line 10 and 12) and comment out the one on line 11.

#LIBS = -Iarmadillo-7.600.2/include -I/opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas -llapack
LIBS = -Iarmadillo-7.600.2/include -framework Accelerate
LINK = -Iarmadillo-7.600.2/include #-I/opt/OpenBLAS/include/

main : $(OBJS)
		$(CC) $(LFLAGS) $(OBJS) -o main $(LIBS)

main.o : main.cpp ShallowNetwork.h IsingDataLoader.h NetworkTrainer.h
		$(CC) $(CFLAGS) main.cpp $(LINK)

ShallowNetwork.o : ShallowNetwork.h ShallowNetwork.cpp NetworkTrainer.h
		$(CC) $(CFLAGS) ShallowNetwork.cpp $(LINK)

IsingDataLoader.o : IsingDataLoader.h IsingDataLoader.cpp
		$(CC) $(CFLAGS) IsingDataLoader.cpp $(LINK)

NetworkTrainer.o : NetworkTrainer.h NetworkTrainer.cpp ShallowNetwork.h
		$(CC) $(CFLAGS) NetworkTrainer.cpp $(LINK)

clean:
		\rm *.o main
