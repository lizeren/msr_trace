#!/usr/bin/env python3
import re
import subprocess
from pathlib import Path


PID = 694760
ADDR = 0x7f1348a33c9b







# === HIT ossl_ess_get_signing_cert pid=694760 tid=694760 comm=openssl ===
# CVE-2026-22796
#         parse_bag+0
#         0x7f1348a33c9b
#         0x7f1348a339c6
#         0x7f1348aaa1e2
#         0x7f1348aa90b0
#         0x7f1348bc9071
#         0x7f13488c86a2
#         0x7f1348bca3b1
#         0x7f1348bca4e2
#         0x7f13488c8ce0
#         0x7f13488c688b
#         0x7f1348bc96b6
#         0x7f1348bc9c8e
#         0x7f1348aa5382
#         0x55adef894b1e
#         0x55adef89313d
#         0x55adef88e942
#         0x55adef847b99
#         0x55adef84770d
#         0x7f13484a4d90








# === HIT ossl_ess_get_signing_cert pid=661484 tid=661484 comm=openssl ===
# CVE-2025-69420
#         ossl_ess_get_signing_cert+0
#         0x7fb06f4f1503
#         0x7fb06f4f1ae0
#         0x7fb06f4f19bb
#         0x563fb75cd3f8
#         0x563fb75cbe0d
#         0x563fb7589b99
#         0x563fb758970d
#         0x7fb06eee3d90


# === HIT CRYPTO_ocb128_encrypt pid=502710 tid=502710 comm=evp_test ===
# CVE-2025-69418
#         CRYPTO_ocb128_encrypt+0
#         0x7fd924ac5f0d
#         0x7fd92489e791
#         0x7fd92489ddd4
#         0x55875fe9ce18
#         0x55875fe9db55
#         0x55875fea9850
#         0x55875feaab3d
#         0x55875feae513
#         0x55875feaf073
#         0x7fd92444dd90


# =========================
# CONFIG (EDIT THESE)
# =========================
# CVE-2025-11187
# PID = 386944
# ADDR = 0x7f39c8a761cb
# ADDR = 0x555e94a7aeb3
# ADDR = 0x555e94a75b99
# ADDR = 0x555e94a7570d
# ADDR = 0x7f39c84e5d90
# =========================


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
