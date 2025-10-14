## br_conditional_count

```bash
sudo sysctl kernel.perf_event_paranoid=1
gcc -O2 -Wall br_conditional_count.c -o br_conditional_count./br_conditional_count
./br_conditional_count 20000000
```

Cross-check with perf stat (alias)

```bash
perf stat -e br_inst_retired.conditional ./br_conditional_count 20000000
```

## br_cond_ntaken_count

```bash
gcc -O2 -Wall -fno-if-conversion -fno-tree-loop-if-convert -fno-tree-vectorize br_cond_ntaken_count.c -o br_cond_ntaken_count
./br_cond_ntaken_count 20000000 0
```

Corss-check

```bash
perf stat -e r10c4 ./br_cond_ntaken_count 20000000 0
```


## br_mispredict_conditional_both

```bash
gcc -O2 -Wall   -fno-if-conversion -fno-tree-loop-if-convert -fno-tree-vectorize   br_mispredict_conditional_both.c -o br_mispredict_conditional_both
./br_mispredict_conditional_both 30000000
```


## l1_miss_hit_cycles_test

```bash
gcc -O2 -Wall l1_miss_hit_cycles_test.c -o l1_miss_hit_cycles_test
./l1_miss_hit_cycles_test
```

## Core Voltage

prerequisites
```bash
sudo modprobe msr # msr-tools
```

### Core Voltage 
pg 4803 Intel 64 and IA-32 Architectures Software Developers Manual

IA32_PERF_STATUS(0x198). Paper Plundervolt also uses this MSR.
https://github.com/KitMurdock/plundervolt/blob/a7313c268d7c27ac3eb806d3ed99019788c5f605/utils/read_voltage.c#L17
```bash
sudo rdmsr 0x198 -u --bitfield 47:32
# or 
echo "scale=2; $(sudo rdmsr 0x198 -u --bitfield 47:32)/8192" | bc
``` 

unhalted core cycles

Table 20-2. Association of Fixed-Function Performance Counters with Architectural Performance Events

fixed counter 1 = unhalted core cycles

refer to rdpmc_demo.c for example usage