## Capture RDPMC trace
From rdpmc_demo.c, we can see even though we can read the RDPMC, the `/sys/kernel/debug/tracing/trace` does not capture the RDPMC trace.
so let's examine how trace works.

## kernel msr driver and tracepoint hooks
From `arch/x86/kernel/msr.c`, we can see msr is a dev that is loaded as the system boot up.
```c
// This implements the read() system call for /dev/cpu/X/msr devices:
static ssize_t msr_read(struct file *file, char __user *buf,
                        size_t count, loff_t *ppos)
```
inside msr_read, it calls rdmsr_safe_on_cpu to read the MSR.
```c
rdmsr_safe_on_cpu(cpu, reg, &data[0], &data[1])
```
It has a definition at `arch/x86/include/asm/msr.h`:
```c
static inline int rdmsr_safe_on_cpu(unsigned int cpu, u32 msr_no,
				    u32 *l, u32 *h)
{
	return rdmsr_safe(msr_no, l, h);
}
```
In the same file msr.h, it has a definition of rdmsr_safe:
```c
/* rdmsr with exception handling */
#define rdmsr_safe(msr, low, high)				\
({								\
	u64 __val;						\
	int __err = native_read_msr_safe((msr), &__val);	\
	(*low) = (u32)__val;					\
	(*high) = (u32)(__val >> 32);				\
	__err;							\
})
```
We can finally find the native_read_msr_safe. `do_trace_read_msr` is hooked whenever a process is trying to read the msr through `/dev/cpu/X/msr`kernel driver.
```c
static inline int native_read_msr_safe(u32 msr, u64 *p)
{
	int err;
	EAX_EDX_DECLARE_ARGS(val, low, high);

	asm volatile("1: rdmsr ; xor %[err],%[err]\n"
		     "2:\n\t"
		     _ASM_EXTABLE_TYPE_REG(1b, 2b, EX_TYPE_RDMSR_SAFE, %[err])
		     : [err] "=r" (err), EAX_EDX_RET(val, low, high)
		     : "c" (msr));
	if (tracepoint_enabled(read_msr))
		do_trace_read_msr(msr, EAX_EDX_VAL(val, low, high), err);

	*p = EAX_EDX_VAL(val, low, high);

	return err;
}
```
## msr-tool
[msr-tools](https://github.com/intel/msr-tools/blob/7d78c80d66463ac598bcc8bf1dc260418788dfda/rdmsr.c#L208) provides the `rdmsr` and `wrmsr` commands at user-space, but it doesn't support `RDPMC`. Let's see how `rdmsr` works so that we can implement our own `RDPMC` utility while let trace capture the RDPMC trace.


from `rdmsr.c` of msr-tools:
```c
void rdmsr_on_cpu(uint32_t reg, int cpu)
{
    sprintf(msr_file_name, "/dev/cpu/%d/msr", cpu);
    fd = open(msr_file_name, O_RDONLY);
    
    // Uses pread() to read at specific position (register number)
    if (pread(fd, &data, sizeof data, reg) != sizeof data) {
        // Error handling
    }
}
```
Opens /dev/cpu/0/msr (or CPU 1, 2, etc.)
Uses pread() which combines lseek + read in one call
The reg parameter becomes the file position
the rest of the call path are explained as above.
```bash
┌─────────────────────────────────────────────────┐
│ Userspace: rdmsr tool                           │
│   pread(fd, buf, 8, 0x10)  // Read MSR 0x10     │
└──────────────────┬──────────────────────────────┘
                   │ System call
                   ↓
┌─────────────────────────────────────────────────┐
│ Kernel: msr.c - msr_read()                      │
│   cpu = iminor(file_inode(file))  // Get CPU #  │
│   reg = *ppos                     // Get MSR #  │
│   rdmsr_safe_on_cpu(cpu, reg, &low, &high)      │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│ Kernel: msr.h - rdmsr_safe_on_cpu()             │
│ [If SMP]:                                       │
│   smp_call_function_single(cpu, ...)            │
│     → Schedules read on target CPU via IPI      │
└──────────────────┬──────────────────────────────┘
                   │ Runs on target CPU
                   ↓
┌─────────────────────────────────────────────────┐
│ Kernel: msr.h - rdmsr_safe()                    │
│   native_read_msr_safe(msr, &val)               │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│ Kernel: msr.h - native_read_msr_safe()          │
│   asm volatile("rdmsr" : outputs : "c"(msr))    │
└──────────────────┬──────────────────────────────┘
                   │ CPU instruction
                   ↓
┌─────────────────────────────────────────────────┐
│ Hardware: x86 CPU                               │
│   RDMSR instruction reads MSR register          │
│   Returns value in EDX:EAX                      │
└─────────────────────────────────────────────────┘
```
Note: in pmcread.c, I use `native_read_pmc()` directly instead of `native_read_msr_safe()` to read the RDPMC.

---

compile and load kernel module
```bash
sudo make
sudo insmod pmcread.ko
ls -l /dev/pmcread
```


compile and run test_pmcread.c
```bash
gcc -O2 -Wall -o test_pmcread test_pmcread.c
# needs root to access /dev/pmcread
sudo ./test_pmcread
```
Every time we run `test_pmcread.c`, we should now be abel to see the RDPMC trace in `/sys/kernel/debug/tracing/trace`.


---

## explanation to `native_read_msr_safe`

<details>
  <summary>Explanation from claude 4.5 sonnet.</summary>

From `arch/x86/include/asm/asm.h` we can see the macro expansion definition of `EAX_EDX_DECLARE_ARGS`, `EAX_EDX_RET`, and `EAX_EDX_VAL`.
```c
// 1. Declare the variables
EAX_EDX_DECLARE_ARGS(val, low, high)
// Expands to: unsigned long low, high;

// 2. Specify outputs in asm statement
EAX_EDX_RET(val, low, high)
// Expands to something like: "=a" (low), "=d" (high)

// 3. Combine low and high into 64-bit value
EAX_EDX_VAL(val, low, high)
// Expands to: ((u64)high << 32 | low)
```


original code:

```c
static inline int native_read_msr_safe(u32 msr, u64 *p)
{
    int err;
    EAX_EDX_DECLARE_ARGS(val, low, high);  // Step 1: Declare variables
    
    asm volatile("1: rdmsr ; xor %[err],%[err]\n"
                 "2:\n\t"
                 _ASM_EXTABLE_TYPE_REG(1b, 2b, EX_TYPE_RDMSR_SAFE, %[err])
                 : [err] "=r" (err), EAX_EDX_RET(val, low, high)  // Step 2: Outputs
                 : "c" (msr));  // Input: ECX = msr number
    
    if (tracepoint_enabled(read_msr))
        do_trace_read_msr(msr, EAX_EDX_VAL(val, low, high), err);  // Step 3: Combine
    
    *p = EAX_EDX_VAL(val, low, high);  // Step 3: Return combined value
    
    return err;
}
```
After Macro Expansion


```c
static inline int native_read_msr_safe(u32 msr, u64 *p)
{
    int err;                    // Error code (0 = success, non-zero = failure)
    unsigned long low, high;    // To receive EDX:EAX from RDMSR
    
    asm volatile(
        "1: rdmsr ; xor %[err],%[err]\n"  // Label 1: RDMSR, then clear err
        "2:\n\t"                           // Label 2: Landing spot if exception
        _ASM_EXTABLE_TYPE_REG(1b, 2b, EX_TYPE_RDMSR_SAFE, %[err])
        : [err] "=r" (err), "=a" (low), "=d" (high)  // Outputs
        : "c" (msr));                                 // Inputs
    
    if (tracepoint_enabled(read_msr))
        do_trace_read_msr(msr, ((u64)high << 32 | low), err);
    
    *p = ((u64)high << 32 | low);  // Store combined 64-bit result
    
    return err;  // Return error code
}

```

## Breaking Down the Assembly

### **The `asm volatile` Syntax**

```c
asm volatile(
    "assembly instructions"
    : output operands
    : input operands
    : clobber list
);

```

### **The Assembly Instructions**

```assembly
"1: rdmsr ; xor %[err],%[err]\n"
"2:\n\t"
_ASM_EXTABLE_TYPE_REG(1b, 2b, EX_TYPE_RDMSR_SAFE, %[err])
```

Let me explain each part:

#### **`1:` - Label 1 (instruction address marker)**

```assembly
1: rdmsr
```

-   `1:` is a **local label** marking the location of the RDMSR instruction
-   Used later for exception handling
-   "If THIS instruction faults, jump to label 2"

#### **`rdmsr` - The actual instruction**

```assembly
rdmsr
```

**What it does:**

-   Reads the MSR specified in ECX register
-   Returns 64-bit value split across EDX:EAX
-   **Can fault** if MSR doesn't exist or isn't accessible

**Register usage:**

```
Input:  ECX = MSR number
Output: EDX:EAX = 64-bit MSR value
        EDX = high 32 bits
        EAX = low 32 bits

```

#### **`; xor %[err],%[err]` - Clear error on success**

```assembly
xor %[err],%[err]

```

-   **XOR** a register with itself = **zero**
-   If RDMSR succeeds, this sets `err = 0` (no error)
-   `%[err]` is a **named operand** (modern GCC syntax)
-   Faster than `mov $0, %[err]`

#### **`2:` - Label 2 (exception landing point)**

```assembly
2:
```

-   This is where execution jumps **if RDMSR faults**
-   At this point, `err` will contain an error code (set by exception handler)
-   Notice there's **no XOR instruction here**, so `err` stays non-zero

#### **`_ASM_EXTABLE_TYPE_REG()` - Exception table entry**

```c
_ASM_EXTABLE_TYPE_REG(1b, 2b, EX_TYPE_RDMSR_SAFE, %[err])

```

This creates an **exception table entry** that tells the kernel:

-   **`1b`**: "back reference to label 1" - the faulting instruction address
-   **`2b`**: "back reference to label 2" - where to jump on fault
-   **`EX_TYPE_RDMSR_SAFE`**: Type of exception handler (sets `err` register)
-   **`%[err]`**: Which register to store the error code in

**How it works internally:**

```c
// Exception table (simplified):
struct exception_table_entry {
    unsigned long insn;     // Address of "1: rdmsr"
    unsigned long fixup;    // Address of "2:"
    int type;               // EX_TYPE_RDMSR_SAFE
    int reg;                // Which register is %[err]
};

```

When RDMSR faults:

1.  CPU triggers general protection fault
2.  Kernel exception handler looks up address of faulting instruction (label 1)
3.  Finds entry in exception table
4.  Sets the `err` register to error code
5.  Jumps to label 2 (skipping the `xor`)
6.  Execution continues normally

### **Output Operands**

```c
: [err] "=r" (err), "=a" (low), "=d" (high)

```

These tell GCC how the assembly outputs map to C variables:

#### **`[err] "=r" (err)`**

-   **`[err]`**: Named operand (can reference as `%[err]` in assembly)
-   **`"=r"`**:
    -   `=` means **write-only output**
    -   `r` means **any general-purpose register** (compiler chooses)
-   **`(err)`**: C variable to store the result

#### **`"=a" (low)`**

-   **`"=a"`**:
    -   `=` write-only output
    -   `a` means **EAX register specifically**
-   **`(low)`**: Store EAX value in `low` variable

#### **`"=d" (high)`**

-   **`"=d"`**:
    -   `=` write-only output
    -   `d` means **EDX register specifically**
-   **`(high)`**: Store EDX value in `high` variable

### **Input Operands**

```c
: "c" (msr)

```

#### **`"c" (msr)`**

-   **`"c"`**: ECX register specifically
-   **`(msr)`**: Load `msr` value into ECX before executing assembly
-   RDMSR reads the MSR number from ECX

## Complete Execution Flow

### **Case 1: Success (MSR exists)**

```
1. ECX ← msr number (0x10)
   
2. Execute: rdmsr
   → CPU reads MSR 0x10
   → EAX ← 0x12345678 (low 32 bits)
   → EDX ← 0xABCDEF00 (high 32 bits)
   
3. Execute: xor %[err], %[err]
   → err ← 0 (success)
   
4. Continue to label 2 (fall through)

5. GCC moves registers to C variables:
   → low = EAX = 0x12345678
   → high = EDX = 0xABCDEF00
   
6. Return err = 0

```

### **Case 2: Failure (MSR doesn't exist)**

```
1. ECX ← msr number (0xBADBAD)
   
2. Execute: rdmsr
   → CPU triggers #GP (General Protection Fault)
   → Kernel exception handler catches it
   
3. Exception handler:
   → Looks up address of "1: rdmsr" in exception table
   → Finds fixup address (label 2)
   → Sets err register to error code (-EIO or similar)
   → Jumps to label 2
   
4. Skip the "xor" instruction (err stays non-zero)

5. Continue execution at label 2

6. Return err = -EIO (or error code)

```

## Visual Comparison

### **Normal Flow (Success):**

```
┌─────────────────────┐
│ 1: rdmsr            │ ✓ Success
├─────────────────────┤
│ xor %[err],%[err]   │ ← Execute this (err = 0)
├─────────────────────┤
│ 2:                  │ ← Fall through to here
├─────────────────────┤
│ (continue...)       │
└─────────────────────┘

```

### **Exception Flow (Failure):**

```
┌─────────────────────┐
│ 1: rdmsr            │ ✗ FAULT!
├─────────────────────┤           ↓ Exception handler
│ xor %[err],%[err]   │ ← SKIP    ↓ sets err = error code
├─────────────────────┤           ↓
│ 2:                  │ ←─────────┘ Jump here
├─────────────────────┤
│ (continue...)       │
└─────────────────────┘

```

## Register Constraints Cheat Sheet

```c
"r" - Any general purpose register (RAX, RBX, RCX, RDX, RSI, RDI, etc.)
"a" - RAX/EAX specifically
"b" - RBX/EBX specifically
"c" - RCX/ECX specifically
"d" - RDX/EDX specifically

"=" - Write-only (output)
"+" - Read-write (input and output)
"&" - Early clobber (written before inputs are read)

"m" - Memory operand
"i" - Immediate constant
"n" - Known constant

```

## Why "Safe"?

Compare to the **unsafe version** (simplified):

```c
static inline u64 __rdmsr(u32 msr)
{
    u32 low, high;
    
    asm volatile("rdmsr"  // No exception handling!
                 : "=a" (low), "=d" (high)
                 : "c" (msr));
    
    return ((u64)high << 32) | low;
}

```

**If you call `__rdmsr()` with invalid MSR:**

-   CPU faults
-   No exception table entry
-   **Kernel panic!** 💥

**With `native_read_msr_safe()`:**

-   CPU faults
-   Exception handler catches it
-   Returns error code
-   Kernel continues running ✅

## Putting It All Together

```c
// You call:
u64 value;
int err = native_read_msr_safe(0x10, &value);

if (err) {
    printk("MSR 0x10 doesn't exist!\n");
} else {
    printk("MSR 0x10 = 0x%llx\n", value);
}

// Assembly executed:
// mov $0x10, %ecx          ← Input: "c" (msr)
// 1: rdmsr                  ← Execute RDMSR
//    xor %edi, %edi         ← Clear err (assuming %edi chosen for err)
// 2:                        ← Exception lands here
// (GCC code to move EAX→low, EDX→high)

```

</details>