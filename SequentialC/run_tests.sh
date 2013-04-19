#!/bin/bash

for i in {8..1}
do
	CFLAGS="-Wall -Wextra -std=c++11 -O3 -ffast-math -DNDEBUG -DTHREADS=${i} -fopenmp"
	LDFLAGS="-lm -fopenmp"
	make clean
	make CXX=g++ CFLAGS="${CFLAGS}" LDFLAGS="${LDFLAGS}"
	/usr/bin/echo -e "\033[1mRunning with ${i} threads.\033[0m"
	n=20
    while [ $(( n -= 1 )) -ge 0 ]
    do
		/usr/bin/time --format=%e ./Simulate < /tmp/stars.in
    done
done
