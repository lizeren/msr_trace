## Install Dependencies

```bash
sudo apt install clang llvm-dev libclang-dev
sudo apt-get install llvm-10-dev clang-10 libclang-10-dev
sudo apt install bear
```

Verify the installation:
```bash
clang --version
llvm-config --version
bear --version
```


## LLVM

### Where could a function call be?

Expression statement: foo();

Assignment statement: x = foo();

Declaration initializer: T x = foo();

Return statement: return foo();

If/while condition: if (foo()) {} / while (foo()) {}

### Step 0: Build the instrumentation tool

```bash
sudo sysctl kernel.perf_event_paranoid=1
mkdir -p build && cd build
cmake ..
make
```

This will:
- Configure the project with CMake (finds LLVM 10)
- Automatically generate `compile_commands.json` for IntelliSense
- Build `build/api_detector`

**Clean build:**
```bash
rm -rf build && mkdir build && cd build && cmake .. && make
```


### Step 1: Create test from reference

We are only allowed to LLVM scan and insert a source file once. So we need to create a test from reference.

```bash
cd test

cp simple_test_ref.c simple_test.c
# Edit simple_test.c as needed
```

### Step 2: Compile test with bear

Using `bear` to generate `compile_commands.json` for LLVM.
```bash
bear gcc -O2 -Wall -g -c simple_test.c
```

### Step 3: Run instrumentation tool
```bash
# From the root directory
./build/api_detector test/simple_test.c \
    --wrap=addition \
    --wrap=subtraction \
    --wrap=multiplication \
    --include-header=pmc.h \
    -- -I. -I/usr/include

# Or from the build directory
cd build
./api_detector ../test/simple_test.c \
    --wrap=addition \
    --wrap=subtraction \
    --wrap=multiplication \
    --include-header=pmc.h \
    -- -I.. -I/usr/include
```

**Note:** The tool uses LLVM 10. Add additional include paths after `--` if needed.


### Print AST of simple_test_ref.c

```bash
clang -Xclang -ast-dump -fsyntax-only test/simple_test_ref.c
```