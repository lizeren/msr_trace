#!/usr/bin/ python3
"""
unique_counters.py
Reads ftrace text and reports unique counter accesses.

- Input trace: autodetected from /sys/kernel/tracing/trace or
  /sys/kernel/debug/tracing/trace
- MSR name decoding: from
  /usr/src/linux-headers-$(uname -r)/arch/x86/include/asm/msr-index.h

Outputs:
  1) Unique RDPMC selectors seen (labeled PMC{n} / FIXED{n}) with counts
  2) Unique MSRs accessed (by read_msr/write_msr), decoded to names, with counts
"""

import os
import re
import sys
import platform
from collections import Counter, defaultdict

# ---------- Paths ----------
TRACE_DIR = "/sys/kernel/debug/tracing/"
TRACE_FILE = os.path.join(TRACE_DIR, "trace")
KREL = platform.uname().release
DEFAULT_MSR_HEADER = f"/usr/src/linux-headers-{KREL}/arch/x86/include/asm/msr-index.h"

# ---------- Load MSR symbols ----------
def load_msr_names(header_path: str):
    msr_names = {}
    try:
        with open(header_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.match(r'#define\s+(MSR_[A-Za-z0-9_]+)\s+(0x[0-9A-Fa-f]+)', line)
                if m:
                    msr_names[int(m.group(2), 16)] = m.group(1)
    except FileNotFoundError:
        pass
    return msr_names

EXTRA_RANGES = (
    ("MSR_LASTBRANCH_%d_FROM_IP", 0x680, 0x69F),
    ("MSR_LASTBRANCH_%d_TO_IP",   0x6C0, 0x6DF),
    ("LBR_INFO_%d",               0xDC0, 0xDDF),
)

def name_msr(addr_hex_str: str, msr_names: dict):
    try:
        n = int(addr_hex_str, 16)
    except ValueError:
        return None
    if n in msr_names: return msr_names[n]
    for templ, lo, hi in EXTRA_RANGES:
        if lo <= n <= hi:
            return templ % (n - lo)
    return None

# ---------- Parse helpers ----------
RDPMC_PAT = re.compile(r'\brdpmc:\s*(?:counter\s*)?([0-9A-Fa-fxX]+)')
MSR_PAT   = re.compile(r'\b((?:read|write)_msr):\s+([0-9A-Fa-f]+)')

def parse_selector(tok: str) -> int:
    t = tok.strip()
    # ftrace usually prints hex without 0x; treat as hex by default
    if t.lower().startswith("0x") or any(c in t.lower() for c in "abcdef"):
        return int(t, 16)
    return int(t, 16)

def label_selector(sel: int) -> str:
    FIXED_BIT = 1 << 30
    if sel & FIXED_BIT:
        return f"FIXED{sel & (FIXED_BIT - 1)}"
    else:
        return f"PMC{sel}"

# ---------- Main ----------
def main():
    msr_header = DEFAULT_MSR_HEADER
    if len(sys.argv) > 1:
        msr_header = sys.argv[1]

    msr_names = load_msr_names(msr_header)

    # containers
    rdpmc_counts = Counter()                 # label -> count
    rdpmc_raw_set = set()                    # raw selector ints
    msr_counts = defaultdict(Counter)        # {'read_msr': Counter(), 'write_msr': Counter()}
    msr_seen = set()

    # read trace file
    try:
        with open(TRACE_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # RDPMC
                mr = RDPMC_PAT.search(line)
                if mr:
                    sel_tok = mr.group(1)
                    try:
                        sel = parse_selector(sel_tok)
                        rdpmc_raw_set.add(sel)
                        rdpmc_counts[label_selector(sel)] += 1
                    except ValueError:
                        pass

                # MSR read/write
                mm = MSR_PAT.search(line)
                if mm:
                    kind = mm.group(1)  # read_msr or write_msr
                    addr = mm.group(2)  # hex (no 0x)
                    msr_counts[kind][addr] += 1
                    msr_seen.add(addr)
    except FileNotFoundError:
        print(f"Trace file not found: {TRACE_FILE}", file=sys.stderr)
        sys.exit(1)

    # ---------- Report ----------
    print(f"# Source trace: {TRACE_FILE}")
    print(f"# MSR header:   {msr_header}")
    print()

    # RDPMC uniques
    if rdpmc_counts:
        print("== Unique RDPMC selectors ==")
        # sort FIXED first then PMC, by index
        def sort_key(lbl):
            if lbl.startswith("FIXED"):
                return (0, int(lbl[5:]))
            if lbl.startswith("PMC"):
                return (1, int(lbl[3:]))
            return (2, lbl)
        for lbl in sorted(rdpmc_counts, key=sort_key):
            print(f"{lbl:>10s}   count={rdpmc_counts[lbl]}")
        print()
    else:
        print("== Unique RDPMC selectors ==\n(none)\n")

    # MSR uniques (grouped by read/write, decode names)
    if msr_seen:
        print("== Unique MSR accesses ==")
        for kind in ("read_msr", "write_msr"):
            if not msr_counts[kind]:
                continue
            print(f"[{kind}]")
            # sort numerically by MSR address
            for addr_hex in sorted(msr_counts[kind], key=lambda h: int(h, 16)):
                name = name_msr(addr_hex, msr_names)
                label = f"{name}({addr_hex})" if name else f"0x{addr_hex}"
                print(f"  {label:<40s} count={msr_counts[kind][addr_hex]}")
            print()
    else:
        print("== Unique MSR accesses ==\n(none)\n")

if __name__ == "__main__":
    main()
