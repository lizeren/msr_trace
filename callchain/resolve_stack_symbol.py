#!/usr/bin/env python3
"""
Resolve a stack frame like "symbol+offset" (e.g., testsuite_test+271)
using nm + addr2line, but if the symbol is not in the main exe, search
other mapped binaries/DSOs from /tmp/maps.<PID>.

Coupled to your workflow:
- /tmp/exe.<PID> contains main executable path
- /tmp/maps.<PID> contains module mappings
Edit only:
- PID
- SYMBOL_OFFSET
"""





"""
=== HIT wc_ChaCha20Poly1305_Encrypt pid=29326 tid=29326 comm=testsuite.test ===
CVE-2025-11931

        wc_ChaCha20Poly1305_Encrypt+0
        wolfcrypt_test+2092
        testsuite_test+271
        main+36
        __libc_start_call_main+128



=== HIT DoTls13CertificateVerify pid=32717 tid=32717 comm=testsuite.test ===
CVE-2025-11934

        DoTls13CertificateVerify+0
        DoTls13HandShakeMsg+663
        DoProcessReplyEx+2271
        ProcessReplyEx+36
        ProcessReply+33
        wolfSSL_connect_TLSv13+730
        wolfSSL_connect+621
        echoclient_test+668
        test_tls+277
        testsuite_test+483
        main+36
        __libc_start_call_main+128
"""


import re
import subprocess
from pathlib import Path

# =========================
# CONFIG (EDIT THESE)
# =========================
PID = 32717
SYMBOL_OFFSET = "DoTls13CertificateVerify+0"   # e.g., "echoclient_test+668"
# =========================

MAPS_PATH = Path(f"/tmp/maps.{PID}")
EXE_PATH  = Path(f"/tmp/exe.{PID}")

SYM_RE = re.compile(r"([A-Za-z0-9_$.@]+)\+([0-9]+)$")
MAP_RE = re.compile(
    r'^([0-9a-f]+)-([0-9a-f]+)\s+'   # start-end
    r'(\S+)\s+'                      # perms
    r'([0-9a-f]+)\s+'                # file offset
    r'\S+\s+\d+\s*'                  # dev inode
    r'(.*)$'                         # path
)

def sh(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

def get_exe() -> str:
    if not EXE_PATH.exists():
        raise SystemExit(f"Missing {EXE_PATH} (did your bpftrace script dump it?)")
    exe = EXE_PATH.read_text().strip()
    if not exe:
        raise SystemExit(f"{EXE_PATH} is empty")
    return exe

def parse_symbol_offset(s: str) -> tuple[str, int]:
    m = SYM_RE.fullmatch(s.strip())
    if not m:
        raise SystemExit(f"Bad SYMBOL_OFFSET format: {s!r}. Expected like 'testsuite_test+271'")
    sym = m.group(1)
    off = int(m.group(2), 10)  # bpftrace ustack offsets are decimal
    return sym, off

def list_modules_from_maps() -> list[str]:
    """
    Return a de-duplicated, ordered list of file-backed modules that have executable mappings.
    We prioritize:
      1) main exe
      2) modules under same tree as exe
      3) everything else
    """
    if not MAPS_PATH.exists():
        raise SystemExit(f"Missing {MAPS_PATH} (did your bpftrace script dump it?)")

    paths = []
    seen = set()

    for line in MAPS_PATH.read_text().splitlines():
        m = MAP_RE.match(line)
        if not m:
            continue
        perms = m.group(3)
        path = m.group(5).strip()

        if "r-x" not in perms:
            continue
        if not path or path.startswith("["):
            continue
        # keep only real files
        p = Path(path)
        if not p.exists():
            continue
        if path not in seen:
            seen.add(path)
            paths.append(path)

    return paths

def nm_symbol_value(binary: str, sym: str) -> int | None:
    """
    Search for exact symbol definition in a binary.
    We try:
      - nm -an (all symbols, sorted by address)
      - if that fails (e.g., huge), fallback to nm -D (dynamic symbols)
    Return integer address if found, else None.
    """
    try:
        out = sh(["nm", "-an", binary])
        for line in out.splitlines():
            parts = line.strip().split()
            # format: <addr> <type> <name>
            if len(parts) >= 3 and parts[2] == sym and parts[0] != "U":
                return int(parts[0], 16)
    except subprocess.CalledProcessError:
        pass

    # fallback: dynamic symbols only (shared libs / stripped exes may still have these)
    try:
        out = sh(["nm", "-D", "-a", binary])
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == sym and parts[0] != "U":
                return int(parts[0], 16)
    except subprocess.CalledProcessError:
        pass

    return None

def addr2line(binary: str, addr: int) -> str:
    return sh(["addr2line", "-e", binary, "-fip", hex(addr)]).strip()

def main():
    exe = get_exe()
    sym, off = parse_symbol_offset(SYMBOL_OFFSET)

    modules = [exe] + [m for m in list_modules_from_maps() if m != exe]

    # Optional heuristic: prefer modules in the same directory tree as the exe
    exe_dir = str(Path(exe).resolve().parent)
    modules.sort(key=lambda p: (0 if p == exe else 1, 0 if str(Path(p).resolve()).startswith(exe_dir) else 2))

    found_in = None
    base = None

    for m in modules:
        val = nm_symbol_value(m, sym)
        if val is not None:
            found_in = m
            base = val
            break

    if found_in is None:
        raise SystemExit(
            f"Symbol {sym!r} not found in exe or any r-xp mapped module from {MAPS_PATH}.\n"
            f"Tip: it may be in a dlopened plugin/provider not present in maps snapshot, or the name differs."
        )

    target = base + off

    print(f"pid: {PID}")
    print(f"symbol: {sym}")
    print(f"offset: {off} (0x{off:x})")
    print(f"found_in: {found_in}")
    print(f"nm base: {hex(base)}")
    print(f"target: {hex(target)}")
    print()
    print("addr2line:")
    print(addr2line(found_in, target))

if __name__ == "__main__":
    main()