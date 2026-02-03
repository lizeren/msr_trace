Find if the function is in the binary
```bash
nm -a /mnt/linuxstorage/openssl/libcrypto.so | grep -w ossl_ess_get_signing_cert
```

Run the uprobe

```bash
sudo bpftrace --unsafe test.bt | tee /tmp/linebuffer_hits.log
```

Run the call chain solver
```bash
python3 resolve_stack.py
```