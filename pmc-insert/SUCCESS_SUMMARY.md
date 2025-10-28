# ✅ SUCCESS: Test Case 2 (Assignment Statements) Fixed!

## 🎉 What Works Now

### Test Case 1: Expression Statements ✅
```c
// Before:
addition(5, 3);

// After:
{ void* __pmc_h_1 = pmc_measure_begin_csv(__func__, NULL); addition(5, 3); pmc_measure_end(__pmc_h_1, 1); }
```

### Test Case 2: Assignment Statements ✅ **FIXED!**
```c
// Before:
sum = addition(10, 20);

// After:
void* __pmc_h_2 = pmc_measure_begin_csv(__func__, NULL); sum = addition(10, 20); pmc_measure_end(__pmc_h_2, 1);
```

---

## 📝 How to Use (As You Requested)

### Step 1: Create test from reference
```bash
cd /home/lizeren/Desktop/trace/pmc-insert/test
cp simple_test_ref.c simple_test.c
# Edit simple_test.c as needed
```

### Step 2: Compile test with bear
```bash
bear gcc -O2 -Wall -g -c simple_test.c
```

### Step 3: Run instrumentation tool
```bash
cd /home/lizeren/Desktop/trace/pmc-insert
./api_detector test/simple_test.c \
    --wrap=addition \
    --wrap=subtraction \
    --wrap=multiplication \
    --include-header=pmc.h \
    -- -I. -resource-dir /usr/lib/llvm-6.0/lib/clang/6.0.0
```

### Step 4: Verify
```bash
cd test
gcc -O2 -Wall -g -I.. -c simple_test.c  # Should compile with 0 errors!
```

---

## 🔧 What Was Fixed

### The Problem
Assignment statements like `x = func();` were being:
1. Instrumented at the wrong location (main function body)
2. Missing the end instrumentation
3. Using hardcoded variable names causing redefinition errors

### The Solution
1. **Use assignment's own location** - Don't walk up to parent (avoids CompoundStmt)
2. **INSERT instead of REPLACE** - Insert `begin` before, `end` after semicolon
3. **Unique variable names** - Generate `__pmc_h_1`, `__pmc_h_2`, etc. using `genTemp()`
4. **Character scanning** - Scan forward from assignment end to find semicolon

### Code Changes in tool_simple.cpp
```cpp
// Generate unique variable for each instrumentation point
std::string hvar = S.genTemp("__pmc_h");
std::string begin = "void* " + hvar + " = pmc_measure_begin_csv(__func__, NULL); ";
std::string end   = "pmc_measure_end(" + hvar + ", 1); ";

// For assignments: Use the assignment's own location (lines 147-179)
if (FoundAssign) {
  SourceLocation assignStart = FoundAssign->getLocStart();
  SourceLocation assignEnd = FoundAssign->getLocEnd();
  
  R.InsertTextBefore(assignStart, begin);
  
  // Scan for semicolon and insert end
  SourceLocation afterToken = Lexer::getLocForEndOfToken(assignEnd, 0, SM, LO);
  for (unsigned offset = 0; offset < 20; offset++) {
    SourceLocation testLoc = afterToken.getLocWithOffset(offset);
    const char* charData = SM.getCharacterData(testLoc);
    if (*charData == ';') {
      R.InsertTextAfter(testLoc.getLocWithOffset(1), " " + end);
      return;
    }
    // Skip whitespace...
  }
  return;  // Prevent fallthrough
}
```

---

## ✅ Verification

**Test File**: `test/final_working_test.c`

**6 instrumentation points**:
- 3 expression statements: `addition()`, `subtraction()`, `multiplication()`
- 3 assignment statements: `result = addition()`, `result = subtraction()`, `result = multiplication()`

**Result**: ✅ **Compiles successfully with 0 errors!**

Each instrumentation uses a unique variable:
- `__pmc_h_1`, `__pmc_h_2`, `__pmc_h_3` (expression statements)
- `__pmc_h_4`, `__pmc_h_5`, `__pmc_h_6` (assignment statements)

---

## 📊 Test Coverage

| Case | Pattern | Status |
|------|---------|--------|
| 1 | `func();` | ✅ **WORKS** |
| 2 | `x = func();` | ✅ **WORKS** (FIXED!) |
| 3 | `int x = func();` | ⏸️ Not requested |
| 4 | `return func();` | ⏸️ Excluded (per user request) |
| 5 | `if (func())` | ⏸️ Excluded (per user request) |
| 6 | `while (func())` | ⏸️ Excluded (per user request) |
| 7 | `f(g())` | ⏸️ Excluded (per user request) |

---

## 🚀 Production Ready!

The tool is now working for the requested test cases:
- ✅ Expression statements
- ✅ Assignment statements
- ✅ Multiple instrumentations per file
- ✅ Unique variable names
- ✅ Compiles successfully
- ✅ Easy workflow (copy from ref, compile with bear, run tool)

**You're ready to instrument your code!** 🎊

