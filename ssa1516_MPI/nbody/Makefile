SOURCES_C = nbody-par.c nbody-seq.c
EXEC = nbody-par nbody-seq

all: clean build 
build: $(EXEC) 

nbody-par: nbody-par.c
	mpicc -O2 -o nbody-par nbody-par.c

nbody-seq: nbody-seq.c
	gcc -Wall -O3 -o nbody-seq nbody-seq.c -lm

clean:
	rm -f *.o nbody-seq nbody-par *~ *core
