# Makefile for testing instrumented code
# This demonstrates how to use -finstrument-functions with libpmc

CC = gcc
# Use -O0 or -fno-inline to prevent function inlining which breaks instrumentation
# -rdynamic exports symbols so dladdr can resolve them from the shared library
CFLAGS = -O0 -Wall -Wextra -finstrument-functions -fno-inline -fplugin=./instrument_attribute.so -fplugin-arg-instrument_attribute-include-function-list=multiplication,helper_func

LDFLAGS = -L.. -lpmc -ldl -lpthread -Wl,-rpath,'$$ORIGIN/..' -rdynamic

# Test program
instrumented_test: instrumented_test.c ../libpmc.so
	@echo "Building instrumented test program..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Run test
run: instrumented_test
	@echo "Running instrumented test..."
	@echo "Make sure api_lists.csv and pmc_events.csv are in the current directory"
	@echo "Set PMC_EVENT_INDICES environment variable, e.g.:"
	@echo "  export PMC_EVENT_INDICES=\"0,1,2\""
	@echo ""
	./instrumented_test

clean:
	rm -f instrumented_test pmc_results.json *.log

.PHONY: run clean

