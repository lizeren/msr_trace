#!/bin/bash
# Wrapper: run openssl in background, capture /proc/<pid>/maps before it exits,
# then wait for it to finish.
#
# Usage: ./run_openssl.sh
# After bpftrace prints the raw stack, use the maps file + probe_runtime_addr
# to resolve symbols offline.

OPENSSL_DIR=/mnt/linuxstorage/openssl

LD_LIBRARY_PATH="$OPENSSL_DIR" \
  "$OPENSSL_DIR/apps/openssl" cms \
  -decrypt -in trigger.cms \
  -recip test/certs/ca-cert.pem \
  -inkey test/certs/ca-key.pem &

PID=$!

# Snapshot maps and exe path as fast as possible while process is alive
cp /proc/$PID/maps /tmp/maps.$PID 2>/dev/null
readlink /proc/$PID/exe > /tmp/exe.$PID 2>/dev/null

wait $PID
STATUS=$?

echo "openssl exited with status $STATUS"
echo "maps: $(wc -l < /tmp/maps.$PID 2>/dev/null) lines -> /tmp/maps.$PID"
echo "exe:  $(cat /tmp/exe.$PID 2>/dev/null)"
