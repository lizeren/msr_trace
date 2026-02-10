#!/usr/bin/env python3
import re
import subprocess
from pathlib import Path


PID = 177970
ADDR = 0x7fe8e5d11791




MAPS_PATH = Path(f"/tmp/maps.{PID}")

MAP_RE = re.compile(
    r'^([0-9a-f]+)-([0-9a-f]+)\s+'   # start-end
    r'(\S+)\s+'                      # perms
    r'([0-9a-f]+)\s+'                # file offset
    r'\S+\s+\d+\s*'                  # dev inode
    r'(.*)$'                         # path
)

def run_addr2line(binary: str, file_off: int) -> str:
    try:
        out = subprocess.check_output(
            ["addr2line", "-e", binary, "-fip", hex(file_off)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return out.strip()
    except Exception as e:
        return f"(addr2line failed) {e}"

def main():
    if not MAPS_PATH.exists():
        raise SystemExit(f"Missing {MAPS_PATH} (did dump_pbmac.bt create it?)")

    candidates = []

    for line in MAPS_PATH.read_text().splitlines():
        m = MAP_RE.match(line)
        if not m:
            continue

        lo = int(m.group(1), 16)
        hi = int(m.group(2), 16)
        perms = m.group(3)
        map_file_off = int(m.group(4), 16)
        path = m.group(5).strip()

        # executable code region
        if "r-x" not in perms:
            continue

        # must contain address
        if not (lo <= ADDR < hi):
            continue

        # skip anonymous mappings
        if not path or path.startswith("["):
            continue

        file_off = map_file_off + (ADDR - lo)
        candidates.append((path, file_off, line))

    if not candidates:
        raise SystemExit(f"No executable mapping found for {hex(ADDR)} in {MAPS_PATH}")

    # Usually exactly one match. If multiple, take the first.
    path, file_off, line = candidates[0]

    print("MAP:", line)
    print("BINARY:", path)
    print("FILE_OFF:", hex(file_off))
    print()
    print("addr2line:")
    print(run_addr2line(path, file_off))

if __name__ == "__main__":
    main()
