# Manual Makefile for building api_detector without CMake
# Uses llvm-config to find proper paths

CXX = g++
LLVM_CONFIG = llvm-config-6.0

# Get LLVM/Clang compile and link flags
LLVM_CXXFLAGS = $(shell $(LLVM_CONFIG) --cxxflags)
LLVM_LDFLAGS = $(shell $(LLVM_CONFIG) --ldflags)
LLVM_LIBS = $(shell $(LLVM_CONFIG) --libs)

# Clang-specific libraries
CLANG_LIBS = -lclangTooling -lclangFrontend -lclangDriver -lclangSerialization \
             -lclangCodeGen -lclangParse -lclangSema -lclangAnalysis \
             -lclangAST -lclangASTMatchers -lclangBasic -lclangEdit \
             -lclangLex -lclangRewrite

# System libraries
SYS_LIBS = -lpthread -ldl -lz

TARGET = api_detector
SRC = tool_simple.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -std=c++17 $(SRC) $(LLVM_CXXFLAGS) $(LLVM_LDFLAGS) $(CLANG_LIBS) $(LLVM_LIBS) $(SYS_LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf build/

test: $(TARGET)
	@echo "Running test suite..."
	$(MAKE) -C test test

.PHONY: test

