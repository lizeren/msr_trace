# Makefile for PMC Library
# 
# Targets:
#   all         - Build shared library and example
#   lib         - Build only the shared library
#   example    - Build example programs
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

# example
example = example_cache_call
EXAMPLE_LDFLAGS = -L. -lpmc -ldl -Wl,-rpath,'$$ORIGIN'



.PHONY: all lib examples clean install help

all: lib example

lib: $(LIB_NAME)

example: $(example)

standalone: $(STANDALONE)

# Build shared library
$(LIB_NAME): $(LIB_OBJS)
	@echo "Building shared library: $@"
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

# Build library object files
%.o: %.c pmc.h
	@echo "Compiling: $<"
	$(CC) $(CFLAGS) -c -o $@ $<

# Build example (linked against shared library)

example_cache_call: example_cache_call.c $(LIB_NAME)
	@echo "Building example: $@"
	$(CC) $(CFLAGS) -o $@ $< $(EXAMPLE_LDFLAGS)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(LIB_NAME) $(LIB_OBJS) $(example) $(STANDALONE)
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
	@echo "  all         - Build shared library and example (default)"
	@echo "  lib         - Build only the shared library"
	@echo "  example    - Build example programs"
	@echo "  clean       - Remove all build artifacts"
	@echo "  install     - Install library system-wide (requires sudo)"
	@echo "  uninstall   - Remove installed library (requires sudo)"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Usage example:"
	@echo "  make                    # Build library and example"
	@echo "  make clean all          # Clean rebuild"
	@echo "  make standalone         # Build original programs"
	@echo "  sudo make install       # Install library system-wide"
	@echo ""
	@echo "Running example:"
	@echo "  ./example_cache_call 100000 sample 100 1000  # Multi-event sampling"

