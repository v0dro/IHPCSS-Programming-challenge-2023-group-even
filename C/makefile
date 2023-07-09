CC=mpicc
CFLAGS=-O0 -Wall -mp=gpu -fopenmp

default: all

all: bin/main

bin/main: src/main.c
	$(CC) -o $@ $< $(CFLAGS)
