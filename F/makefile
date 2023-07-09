CC=mpifort
CFLAGS=-O3 -Wall -mp=gpu -fopenmp

default: all

all: bin/main

bin/main: src/main.f08
	$(CC) -o $@ $< $(CFLAGS)
