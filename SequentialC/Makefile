EXEC=Simulate
CFLAGS=-Wall -Wextra -std=c++11

all: build

build: main.o
	$(CXX) $(LDFLAGS) main.o -o $(EXEC)

main.o: main.cc
	$(CXX) $(CFLAGS) -c main.cc

clean:
	$(RM) *.o $(EXEC)
