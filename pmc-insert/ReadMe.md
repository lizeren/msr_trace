## Install Dependencies

```bash
sudo apt install clang llvm-dev libclang-dev
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