# Attack Gadgets Proof of Concept - Minimal Changes

This directory contains two C programs demonstrating the difference between secure code and vulnerable code with **minimal structural changes**.

## Key Design Principle

The vulnerable program is **nearly identical** to the normal program, with only security checks removed:
- **Minimal architectural impact**: Same function signatures, same return types, same overall structure
- **Error handling via return values**: Functions return 0 for success, -1 for failure
- **Error messages outside functions**: Validation errors printed in main(), not inside functions
- **Only validation removed**: No extra code added, only the `if (condition) { return -1; }` checks deleted

## Programs

### 1. `normal_program.c` - Secure Implementation
Five functions with proper security measures:
- Function 1: String processing with bounds checking
- Function 2: Array access with index validation  
- Function 3: Memory allocation with size validation
- Function 4: File path handling with length validation
- Function 5: Integer parsing with character validation

### 2. `vulnerable_program.c` - Attack Gadgets (Security Checks Removed)
**Identical structure** but with validation removed:

#### Function 1: Buffer Overflow
```diff
- // Bounds checking - prevent buffer overflow
- if (input_len >= MAX_BUFFER) {
-     return -1;
- }
```
**Impact**: Without this check, `strcpy()` can overflow the buffer.

#### Function 2: Out-of-Bounds Array Access
```diff
- // Validate count bounds
- if (count > MAX_ITEMS || count < 0) {
-     return -1;
- }
- // Validate index bounds
- if (index < 0 || index >= MAX_ITEMS) {
-     return -1;
- }
```
**Impact**: Without validation, `local_array[index]` can read beyond array bounds, causing information leakage or crashes.

#### Function 3: Heap Overflow
```diff
- // Validate allocation size
- if (size == 0 || size > 1024) {
-     return -1;
- }
...
- // Safe fill operation with bounds checking
  size_t fill_len = strlen(fill_data);
- if (fill_len >= size) {
-     fill_len = size - 1;
- }
```
**Impact**: Without the length check, `memcpy()` can write beyond allocated buffer size, corrupting heap metadata.

#### Function 4: Path Concatenation Overflow
```diff
- // Check combined length
- if (total_len >= MAX_BUFFER) {
-     return -1;
- }
```
**Impact**: Without length validation, `sprintf()` can overflow the `fullpath` buffer.

#### Function 5: Invalid Input Handling
```diff
- // Validate input
- if (str == NULL || strlen(str) == 0) {
-     return -1;
- }
- // Check for valid integer characters
- for (size_t i = 0; str[i] != '\0'; i++) {
-     if (i == 0 && str[i] == '-') continue;
-     if (str[i] < '0' || str[i] > '9') {
-         return -1;
-     }
- }
```
**Impact**: Without validation, NULL pointer or invalid characters can cause undefined behavior or incorrect parsing.

## Compilation

```bash
# Build both programs
make all

# Build only normal (secure) version
make normal

# Build only vulnerable version
make vulnerable

# Clean up
make clean
```

## Compilation Flags

### Normal Program (Protected)
- `-fstack-protector-strong`: Stack canary protection
- `-D_FORTIFY_SOURCE=2`: Buffer overflow detection
- `-pie -fPIE`: Position Independent Executable (ASLR)

### Vulnerable Program (Unprotected)
- `-fno-stack-protector`: Disable stack canaries
- `-z execstack`: Make stack executable
- `-no-pie`: Disable ASLR for easier exploitation

## Testing

```bash
# Test normal program (all inputs validated)
make test_normal
./normal_program

# Test vulnerable program (with safe inputs)
make test_vulnerable
./vulnerable_program
```

## Differences Summary

| Aspect | Lines Changed |
|--------|---------------|
| Function 1 | 3 lines removed (bounds check) |
| Function 2 | 8 lines removed (count + index validation) |
| Function 3 | 4 lines removed (size validation + length check) |
| Function 4 | 3 lines removed (length check) |
| Function 5 | 9 lines removed (null check + character validation) |
| Architecture | Return values for error handling (0 = success, -1 = fail) |
| **Total** | **~27 lines of validation removed** |

## Exploitation Examples

### Example 1: Buffer Overflow (Function 1)
```bash
# Normal program - rejected
./normal_program <<< "$(python3 -c 'print("A"*100)')"

# Vulnerable program - crashes
./vulnerable_program <<< "$(python3 -c 'print("A"*100)')"
```

### Example 2: Out-of-Bounds Read (Function 2)
```c
// Modify main() to test:
array_processor(test_data, 100, 50);  // Reads beyond array bounds
```

### Example 3: Heap Overflow (Function 3)
```c
// Modify main() to test:
memory_allocator(4, "This string is much longer than 4 bytes");
```

## Educational Purpose

This demonstrates that:

1. **Small changes = Big impact**: Removing ~40 lines creates multiple serious vulnerabilities
2. **Minimal differences**: Easy to compare with diff tools
3. **Real-world relevance**: Missing validation is a common vulnerability pattern
4. **Defense importance**: Every check serves a purpose

## Comparing the Programs

```bash
# See exact differences
diff -u normal_program.c vulnerable_program.c

# Count lines changed
diff normal_program.c vulnerable_program.c | grep "^[<>]" | wc -l
```

## Warning

**FOR EDUCATIONAL AND SECURITY RESEARCH ONLY**

- Do not use these techniques on systems you don't own
- These vulnerabilities are intentional for learning purposes
- Always follow responsible disclosure practices

## References

- OWASP Buffer Overflow: https://owasp.org/www-community/vulnerabilities/Buffer_Overflow
- CWE-119: Improper Restriction of Operations within Memory Bounds
- CWE-120: Buffer Copy without Checking Size of Input
- Secure Coding: https://wiki.sei.cmu.edu/confluence/display/c/SEI+CERT+C+Coding+Standard
