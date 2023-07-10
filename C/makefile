CC=mpicc
CFLAGS=-O2 -Wall -mp=gpu -fopenmp -lm

default: all

all: bin/main

bin/main: src/main.c
	$(CC) -o $@ $< $(CFLAGS)
