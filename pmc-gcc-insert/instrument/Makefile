# Makefile for GCC plugin-based instrumentation

# Detect GCC plugin directory
GCC_PLUGIN_DIR := $(shell gcc -print-file-name=plugin)

# Compiler settings
CXX := g++
# -fno-rtti is required to avoid undefined symbol errors for typeinfo (e.g., _ZTI8opt_pass)
CXXFLAGS := -shared -fPIC -O2 -std=c++17 -Wall -Wextra -fno-rtti
INCLUDES := -I$(GCC_PLUGIN_DIR)/include

# Plugin target
PLUGIN := instrument_callsites_plugin.so
PLUGIN_SRC := insert.cc

.PHONY: all clean plugin test

all: plugin

plugin: $(PLUGIN)

$(PLUGIN): $(PLUGIN_SRC)
	@echo "Building GCC plugin..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
	@echo "Plugin built: $(PLUGIN)"
	@echo ""
	@echo "Usage:"
	@echo "  gcc -fplugin=./$(PLUGIN) \\"
	@echo "      -fplugin-arg-$(basename $(PLUGIN))-include-function-list=func1,func2 \\"
	@echo "      your_program.c -o your_program"

clean:
	rm -f $(PLUGIN)
	@echo "Cleaned plugin"

# Test target (requires test program)
test: plugin
	@echo "Testing plugin..."
	@if [ -f ../test/instrumented_test.c ]; then \
		gcc -O2 \
			-fplugin=./$(PLUGIN) \
			-fplugin-arg-$(basename $(PLUGIN))-debug \
			-fplugin-arg-$(basename $(PLUGIN))-include-function-list=target_function \
			-I.. \
			-L.. \
			-lpmc \
			../test/instrumented_test.c -o ../test/instrumented_test && \
		echo "Test program built: ../test/instrumented_test"; \
	else \
		echo "Test file not found: ../test/instrumented_test.c"; \
	fi

help:
	@echo "GCC Plugin Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the plugin (default)"
	@echo "  plugin  - Build the plugin"
	@echo "  test    - Build plugin and test program"
	@echo "  clean   - Remove built files"
	@echo "  help    - Show this help"
	@echo ""
	@echo "Plugin: $(PLUGIN)"
	@echo "GCC Plugin Dir: $(GCC_PLUGIN_DIR)"
