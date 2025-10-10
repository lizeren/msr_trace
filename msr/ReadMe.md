## Performance counters
This page keeps track of performance counters we may use for tracing

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

### Core Voltage 
unhalted core cycles

Table 20-2. Association of Fixed-Function Performance Counters with Architectural Performance Events

fixed counter 1 = unhalted core cycles

refer to rdpmc_demo.c for example usage