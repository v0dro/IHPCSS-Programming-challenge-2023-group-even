CC=mpicc
CFLAGS=-O2 -Wall -mp=gpu -lm -fopenmp
# CFLAGS=-O2 -Wall -lm -I"$(shell brew --prefix libomp)/include" -L"$(shell brew --prefix libomp)/lib" -lomp
BIN_DIR=bin

default: all

all: create_directory \
	 $(BIN_DIR)/main

create_directory:
	@mkdir -p $(BIN_DIR)

bin/main: src/main.c
	$(CC) -o $@ $< $(CFLAGS)
