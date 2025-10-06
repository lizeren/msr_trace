##  Linux ftrace



### ftrace control interface
`/sys/kernel/debug/tracing/events/` is the ftrace control interface

Each subdirectory represents a tracing subsystem, `msr/` contains events related to MSR operations

Enable tracing by turning tracing on (1) or off (0)

go to `/sys/kernel/debug/tracing/events/msr` to enable all msr events tracing
```bash
echo 1 | sudo tee /sys/kernel/debug/tracing/events/msr/enable
# or we can enable sub-events of msr individually
echo 1 | sudo tee /sys/kernel/debug/tracing/events/msr/read_msr/enable
echo 1 | sudo tee /sys/kernel/debug/tracing/events/msr/write_msr/enable
echo 1 | sudo tee /sys/kernel/debug/tracing/events/msr/rdpmc/enable
```
Not we can read the trace and capture the msr events
```bash
sudo cat /sys/kernel/debug/tracing/trace
```

To clean the trace, write empty string to the trace file
```bash
sudo tee /sys/kernel/debug/tracing/trace < /dev/null
```
### Deconde the trace
The script can print human readable msr name by decoding hex value using the msr-index.h
```bash
sudo cat /sys/kernel/debug/tracing/trace | python3 decode_msr.py /usr/src/linux-headers-$(uname -r)/arch/x86/include/asm/msr-index.h
```
To see the unique counters
```bash
sudo python3 unique_counters.py
```
example
```bash
# clean the trace
echo 1 | sudo tee /sys/kernel/debug/tracing/events/msr/read_msr/enable
sudo tee /sys/kernel/debug/tracing/trace < /dev/null
# https://github.com/torvalds/linux/blob/7a405dbb0f036f8d1713ab9e7df0cd3137987b07/arch/x86/include/asm/msr-index.h#L1183
sudo rdmsr 0x30a              # Read fixed counter 1 MSR
# you will read ffffcabcacaa
sudo cat /sys/kernel/debug/tracing/trace | python3 decode_msr.py /usr/src/linux-headers-$(uname -r)/arch/x86/include/asm/msr-index.h
```
```bash
## result, which matches the result of sudo rdmsr 0x30a
<idle>-0       [000] d.h.  4792.007564: read_msr: MSR_CORE_PERF_FIXED_CTR1(30a), value ffffcabcacaa
```


---

### rdpmc in user-space
#### prerequisite
```bash
sudo apt install -y msr-tools # provides the rdmsr and wrmsr commands
```
compile and run rdpmc_demo.c
```bash
gcc -O0 -Wall rdpmc_demo.c -o rdpmc_demo
./rdpmc_demo
```
Running rdpmc_demo.c will give you segmentation fault because the kernel hasn’t allowed user-mode RDPMC. On x86, 
a disallowed RDPMC raises a #GP fault, which the OS delivers to your process as SIGSEGV
we can check the permission of user-mode RDPMC and the strictness of perf
```bash
# 0 = disallowed, 1 = allowed to the event owner, 2 = allowed to any user
cat /sys/bus/event_source/devices/cpu/rdpmc

# How strict perf is (lower is more permissive; -1 is most permissive; 3 is most strict)
cat /proc/sys/kernel/perf_event_paranoid
```
Permit user-mode RDPMC globally
```bash
# allow everyone to use RDPMC (2), or 1 to allow owner
echo 2 | sudo tee /sys/bus/event_source/devices/cpu/rdpmc

# make perf more permissive (optional)
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```


### Enable fixed counter 0 (instructions retired) on CPU 0
Before we run rdpmc_demo.c, we need to enable the fixed counter 0 and count the instructions retired
```bash
#Load driver once
sudo modprobe msr

# 1) Enable fixed0 in user+kernel by OR-ing in 0x3 to the low nibble
#    Current 0x38D = 0xB0 (ctr1 nibble=0xB, ctr0 nibble=0x0)
#    we can check by
sudo rdmsr -p 0 0x38d
#    New            = 0xB0 | 0x03 = 0xB3
sudo wrmsr -p 0 0x38d 0xb3

# 2) (Optional) clear fixed0 to start fresh
sudo wrmsr -p 0 0x309 0

# 3) Sanity: read both counters
sudo rdmsr -p 0 0x309   # IA32_FIXED_CTR0 (should start increasing)
sudo rdmsr -p 0 0x30a   # IA32_FIXED_CTR1 (already increasing)
```


BR_INST_RETIRED.ALL_BRANCHES Counts all (macro) branch instructions retired. Errata: SKL091 EventSel=C4H UMask=00H Counter=0,1,2,3 CounterHTOff=0,1,2,3,4,5,6,7 Architectural, AtRetirement

https://perfmon-events.intel.com/index.html?pltfrm=skylake.html&evnt=BR_INST_RETIRED.ALL_BRANCHES


```bash
sudo modprobe msr

# Disable counters while programming
# Figure 20-3. Layout of IA32_PERF_GLOBAL_CTRL MSR
sudo wrmsr -p 0 0x38F 0                      # IA32_PERF_GLOBAL_CTRL

# Program IA32_PERFEVTSEL0 (0x186):
# Figure 20-1. Layout of IA32_PERFEVTSELx MSRs
# USR=1 (bit16), OS=1 (bit17), EN=1 (bit22), event=0xC4, umask=0x00
# PERFEVTSEL = (umask<<8)|event|(1<<16)|(1<<17)|(1<<22) = 0x004300C4
sudo wrmsr -p 0 0x186 0x004300C4

# Clear the counter
sudo wrmsr -p 0 0x0C1 0                      # IA32_PMC0

# Enable PMC0 (bit 0) globally
sudo wrmsr -p 0 0x38F 0x1
```


