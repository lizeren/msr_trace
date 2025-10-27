# Makefile for PMC Library
# 
# Targets:
#   all         - Build shared library and examples
#   lib         - Build only the shared library
#   examples    - Build example programs
#   clean       - Remove build artifacts
#   install     - Install library (requires sudo)

CC = gcc
CFLAGS = -O2 -Wall -Wextra -fPIC
LDFLAGS = -shared
LIBS = -ldl

# Library
LIB_NAME = libpmc.so
LIB_SRCS = pmc.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

# Examples
EXAMPLES = example_usage example_l1_cache example_cache_call
EXAMPLE_LDFLAGS = -L. -lpmc -ldl -Wl,-rpath,'$$ORIGIN'

# Standalone programs (for comparison)
STANDALONE = near_call_counter_sample near_call_counter \
             l1_miss_hit_cycles_test br_mispredict_conditional_both \
             br_cond_ntaken_count br_conditional_count

.PHONY: all lib examples standalone clean install help

all: lib examples

lib: $(LIB_NAME)

examples: $(EXAMPLES)

standalone: $(STANDALONE)

# Build shared library
$(LIB_NAME): $(LIB_OBJS)
	@echo "Building shared library: $@"
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

# Build library object files
%.o: %.c pmc.h
	@echo "Compiling: $<"
	$(CC) $(CFLAGS) -c -o $@ $<

# Build examples (linked against shared library)
example_usage: example_usage.c $(LIB_NAME)
	@echo "Building example: $@"
	$(CC) $(CFLAGS) -o $@ $< $(EXAMPLE_LDFLAGS)

example_l1_cache: example_l1_cache.c $(LIB_NAME)
	@echo "Building example: $@"
	$(CC) $(CFLAGS) -o $@ $< $(EXAMPLE_LDFLAGS)

example_cache_call: example_cache_call.c $(LIB_NAME)
	@echo "Building example: $@"
	$(CC) $(CFLAGS) -o $@ $< $(EXAMPLE_LDFLAGS)

# Build standalone programs
near_call_counter_sample: near_call_counter_sample.c
	@echo "Building standalone: $@"
	$(CC) $(CFLAGS) -o $@ $< -ldl

near_call_counter: near_call_counter.c
	@echo "Building standalone: $@"
	$(CC) $(CFLAGS) -o $@ $< -ldl

l1_miss_hit_cycles_test: l1_miss_hit_cycles_test.c
	@echo "Building standalone: $@"
	$(CC) $(CFLAGS) -o $@ $<

br_mispredict_conditional_both: br_mispredict_conditional_both.c
	@echo "Building standalone: $@"
	$(CC) $(CFLAGS) -o $@ $<

br_cond_ntaken_count: br_cond_ntaken_count.c
	@echo "Building standalone: $@"
	$(CC) $(CFLAGS) -o $@ $<

br_conditional_count: br_conditional_count.c
	@echo "Building standalone: $@"
	$(CC) $(CFLAGS) -o $@ $<

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(LIB_NAME) $(LIB_OBJS) $(EXAMPLES) $(STANDALONE)
	rm -f *.o

# Install library system-wide (requires sudo)
install: $(LIB_NAME)
	@echo "Installing $(LIB_NAME) to /usr/local/lib..."
	install -m 755 $(LIB_NAME) /usr/local/lib/
	@echo "Installing pmc.h to /usr/local/include..."
	install -m 644 pmc.h /usr/local/include/
	@echo "Running ldconfig..."
	ldconfig
	@echo "Installation complete!"

# Uninstall library
uninstall:
	@echo "Uninstalling PMC library..."
	rm -f /usr/local/lib/$(LIB_NAME)
	rm -f /usr/local/include/pmc.h
	ldconfig
	@echo "Uninstall complete!"

# Help target
help:
	@echo "PMC Library Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build shared library and examples (default)"
	@echo "  lib         - Build only the shared library"
	@echo "  examples    - Build example programs"
	@echo "  standalone  - Build original standalone programs"
	@echo "  clean       - Remove all build artifacts"
	@echo "  install     - Install library system-wide (requires sudo)"
	@echo "  uninstall   - Remove installed library (requires sudo)"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                    # Build library and examples"
	@echo "  make clean all          # Clean rebuild"
	@echo "  make standalone         # Build original programs"
	@echo "  sudo make install       # Install library system-wide"
	@echo ""
	@echo "Running examples:"
	@echo "  ./example_usage                    # Count mode, 1M iterations"
	@echo "  ./example_usage 5000000            # Count mode, 5M iterations"
	@echo "  ./example_usage 5000000 sample     # Sample mode"
	@echo "  ./example_l1_cache                 # L1 cache behavior analysis"
	@echo "  ./example_cache_call 100000 sample 100 1000  # Multi-event sampling"

