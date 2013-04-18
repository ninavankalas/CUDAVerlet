
EXEC=Simulate
CC=$(shell which nvcc)
CFLAGS=-arch=compute_30 -code=sm_30 -O3

.PHONY = all cuda clean

ifeq (${DEBUG},y)
CFLAGS += -g -G --ptxas-options=-v
endif

all: cuda

cuda: main.o
	$(CC) $(LDFLAGS) main.o -o $(EXEC) 

main.o: main.cu
	$(CC) $(CFLAGS) -c main.cu
	
clean:
	$(RM) *.o $(EXEC)
