# go to /sys/kernel/debug/tracing/events/msr
# enable all msr events tracing
echo 1 | sudo tee /sys/kernel/debug/tracing/events/msr/enable
# or we can enable sub-events of msr individually
echo 1 | sudo tee /sys/kernel/debug/tracing/events/msr/read_msr/enable
echo 1 | sudo tee /sys/kernel/debug/tracing/events/msr/write_msr/enable
echo 1 | sudo tee /sys/kernel/debug/tracing/events/msr/rdpmc/enable

# decode the trace
sudo cat /sys/kernel/debug/tracing/trace | python3 decode_msr.py /usr/src/linux-headers-$(uname -r)/arch/x86/include/asm/msr-index.h

sudo apt install -y msr-tools # provides the rdmsr and wrmsr commands

# Enable fixed counter 0 (instructions retired) on CPU 0
# Load driver once
sudo modprobe msr

# 1) Enable fixed0 in user+kernel by OR-ing in 0x3 to the low nibble
#    Current 0x38D = 0xB0 (ctr1 nibble=0xB, ctr0 nibble=0x0)
#    New            = 0xB0 | 0x03 = 0xB3
sudo wrmsr -p 0 0x38d 0xb3

# 2) (Optional) clear fixed0 to start fresh
sudo wrmsr -p 0 0x309 0

# 3) Sanity: read both counters
sudo rdmsr -p 0 0x309   # IA32_FIXED_CTR0 (should start increasing)
sudo rdmsr -p 0 0x30a   # IA32_FIXED_CTR1 (already increasing)


# compile rdpmc_demo.c
gcc -O0 -Wall rdpmc_demo.c -o rdpmc_demo
# if running rdpmc_demo.c gives you segmentation fault, the kernel hasn’t allowed user-mode RDPMC. On x86, 
# a disallowed RDPMC raises a #GP fault, which the OS delivers to your process as SIGSEGV

# 0 = disallowed, 1 = allowed to the event owner, 2 = allowed to any user
cat /sys/bus/event_source/devices/cpu/rdpmc

# How strict perf is (lower is more permissive; -1 is most permissive; 3 is most strict)
cat /proc/sys/kernel/perf_event_paranoid

#Permit user-mode RDPMC globally
# allow everyone to use RDPMC (2), or 1 to allow owner
echo 2 | sudo tee /sys/bus/event_source/devices/cpu/rdpmc

# make perf more permissive (optional)
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid




# Example uses PMC0 on CPU 0. This counts in user and kernel.

# BR_INST_RETIRED.ALL_BRANCHES Counts all (macro) branch instructions retired. Errata: SKL091 EventSel=C4H UMask=00H Counter=0,1,2,3 CounterHTOff=0,1,2,3,4,5,6,7 Architectural, AtRetirement
# https://perfmon-events.intel.com/index.html?pltfrm=skylake.html&evnt=BR_INST_RETIRED.ALL_BRANCHES


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
