# Simple Makefile for PMC instrumentation testing
# Usage:
#   make ref    - Prepare test file (what ref.bash does)
#   make run    - Compile and run instrumented code

.PHONY: ref run clean

# Target 1: What ref.bash does - prepare for LLVM instrumentation
ref:
	rm -f simple_test.c
	rm -f simple_test.o
	cp simple_test_ref.c simple_test.c
	bear gcc -O2 -Wall -g -c simple_test.c

# Target 2: Compile and run the instrumented code with PMC library
run:
	gcc -O2 -Wall -g simple_test.c -L. -lpmc -Wl,-rpath,. -o simple_test
	LD_LIBRARY_PATH=. ./simple_test

# Clean generated files
clean:
	rm -f simple_test.c simple_test.o simple_test compile_commands.json

