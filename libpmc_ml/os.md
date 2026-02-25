# OS-level factors

Which OS-level factors cause the biggest signature shifts

## 1. Governor / turbo


```bash
# Quick overview (all CPUs)
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# If you see mostly powersave → frequency ramps conservatively
# If you see performance → CPU stays at max frequency

# Set performance
sudo cpupower frequency-set -g performance

# Set powersave
sudo cpupower frequency-set -g powersave
```


# 2. default lazy binding vs eager binding

LD_BIND_NOW is not parsed as a boolean 0/1. The rule is:

If LD_BIND_NOW is set to a nonempty string → immediate binding (resolve symbols at startup).

If it is unset → default behavior (lazy binding for functions)

```bash
LD_BIND_NOW=1 ./your_program [args]
```

observation: by default lazy binding is used, if eager binding is used, the accuracy will drop a lot

# 3. pin to specific CPU

```bash
taskset -c 0 ./your_program [args]
```
observation: looks like pinning to specific CPU will not affect the accuracy

# 4. ASLR

```bash
# Check current ASLR setting
cat /proc/sys/kernel/randomize_va_space
# 0 = off
# 1 = conservative randomization
# 2 = full randomization (most distros default)



# Turn ASLR off (until reboot or you change it back)
sudo sysctl -w kernel.randomize_va_space=0
# Turn ASLR back on (typical default)
sudo sysctl -w kernel.randomize_va_space=2
```

observation: looks like ASLR will not affect the accuracy