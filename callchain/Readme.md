## openssl
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
python3 resolve_stack_ip.py
```


## wolfssl

Find if the function is in the binary
```bash
nm -D --defined-only /mnt/linuxstorage/wolfssl/src/.libs/libwolfssl.so | grep -w wc_ChaCha20Poly1305_Encrypt
```
Run the uprobe

```bash
sudo bpftrace --unsafe test.bt | tee /tmp/linebuffer_hits.log
```

unlike openssl that only gives you the raw instruction addresses, wolfssl shows you the callee function by using the caller function name plus the offset relative to the caller.
```bash
wc_ChaCha20Poly1305_Encrypt+0
        wolfcrypt_test+2092
        testsuite_test+271
        main+36
        __libc_start_call_main+128
```
To find the address of the function

```bash
# 1. Get the test executable:
exe=$(cat /tmp/exe.25937)
echo "$exe"
# 2. Get the symbol value of testsuite_test in that executable:
nm -an "$exe" | grep -w 'testsuite_test$'
# Example output:
00000000000012340 T testsuite_test
# 3. Add 271 bytes (decimal) to that symbol value:

271 decimal = 0x10f hex

target = 0x12340 + 0x10f = 0x1244f

# 4. Resolve with addr2line:
addr2line -e "$exe" -fip 0x1244f
```


Run the call chain solver for wolfssl
```bash
python3 resolve_stack_symbol.py
```