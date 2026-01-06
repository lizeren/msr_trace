# Makefile for PMC Library
# 
# Targets:
#   all         - Build shared library
#   lib         - Build only the shared library
#   clean       - Remove build artifacts
#   install     - Install library (requires sudo)

CC = gcc
CFLAGS = -O2 -Wall -Wextra -fPIC
LDFLAGS = -shared
LIBS = -ldl -lpthread

# For instrumentation: add -finstrument-functions when compiling your application
# Example: gcc -finstrument-functions -o myapp myapp.c -L. -lpmc -ldl

# Library
LIB_NAME = libpmc.so
LIB_SRCS = pmc.c
LIB_OBJS = $(LIB_SRCS:.c=.o)



.PHONY: all lib clean install help

all: lib

lib: $(LIB_NAME)

# Build shared library
$(LIB_NAME): $(LIB_OBJS)
	@echo "Building shared library: $@"
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

# Build library object files
%.o: %.c pmc.h
	@echo "Compiling: $<"
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(LIB_NAME) $(LIB_OBJS)
	rm -f *.o
	rm -f pmc_results.json

# Help target
help:
	@echo "PMC Library Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build shared library (default)"
	@echo "  lib         - Build only the shared library"
	@echo "  clean       - Remove all build artifacts"
	@echo "  install     - Install library system-wide (requires sudo)"
	@echo "  uninstall   - Remove installed library (requires sudo)"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Usage example:"
	@echo "  make                    # Build library"
	@echo "  make clean all          # Clean rebuild"
	@echo "  sudo make install       # Install library system-wide"

