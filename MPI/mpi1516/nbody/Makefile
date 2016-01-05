SOURCES_C = nbody-par.c nbody-seq.c nbody-par_other_method.c nbody-par_other_method2.c nbody-par_other_method_complete.c
EXEC = nbody-par nbody-seq nbody-par_other_method nbody-par_other_method2 nbody-par_other_method_complete

build: $(EXEC) 
all: clean build 

nbody-par: nbody-par.c
	mpicc -O2 -o nbody-par nbody-par.c

nbody-par_other_method: nbody-par_other_method.c
	mpicc -O2 -o nbody-par_other_method nbody-par_other_method.c

nbody-par_other_method2: nbody-par_other_method2.c
	mpicc -O2 -o nbody-par_other_method2 nbody-par_other_method2.c

nbody-par_other_method_complete: nbody-par_other_method_complete.c
	mpicc -O2 -o nbody-par_other_method_complete nbody-par_other_method_complete.c

nbody-seq: nbody-seq.c
	gcc -Wall -O3 -o nbody-seq nbody-seq.c -lm

clean:
	rm -f nbody-par.o nbody-seq.o nbody-par_other_method.o nbody-par_other_method2.o nbody-par_other_method_complete.o  program core *~
