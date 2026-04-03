## WolfSSL

***Important: WolfSSL version: git checkout v5.7.2-stable***


### TL;DR

We provide a folder of all the different builds. you can just compile them and run the collector script by:

```bash
bash compile.sh
bash collector-inter.sh
bash collector-intra.sh
```
### Pre-requirement: release perf_event_paranoid

If you don't get any results from libpmc measurement, always run this command and will most likely fix the problem.

```bash
sudo sysctl kernel.perf_event_paranoid=1
export LD_BIND_NOW=1
```


### place to put libpmc.so, pmc.h, collect_pmc_features.py and pmc_events.csv

libpmc.so: wolfssl/testsuite/libpmc.so
pmc.h: wolfssl/testsuite/pmc.h
collect_pmc_features.py: wolfssl/collect_pmc_features.py
pmc_events.csv: wolfssl/pmc_events.csv

```bash
./autogen.sh
# compile shared library version
./configure --enable-all LIBS="-L$PWD/testsuite -lpmc -lcontext_mixer"
# compile static library version
./configure --disable-shared --enable-static --enable-all 
# This compiles to O0. Without this specification it defaults to O2.
make CFLAGS="-O0" LIBS="-L$PWD/testsuite -lpmc -lcontext_mixer" -j$(nproc)

export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"
# run the test program
export MIXER_INDICES=1 && export PMC_EVENT_INDICES="0,1,2,3" && LD_LIBRARY_PATH="./src/.libs:./testsuite/.libs:./:./" ./testsuite/.libs/testsuite.test

# why not just invoke ./testsuite/testsuite.test?
# ./testsuite/testsuite.test is a script not a binary. During cross OS version measurement, we need to link against custom Glibc. The best way to do this is to use the binary directly.

# python collector

LD_LIBRARY_PATH="./src/.libs:./testsuite/.libs:./:./" python3 collect_pmc_features_mixer.py --target "LD_LIBRARY_PATH="./src/.libs:./testsuite/.libs:./:./" ./testsuite/.libs/testsuite.test" --runs 5 --total 1 --name wolfssl --start 1 &> /dev/null
```


`wc_Sha256Update`
`wc_Sha256GetHash`
`wc_Sha256Copy`
`wc_ShaUpdate`
`wc_ShaGetHash`
`wc_ShaCopy`
`wc_InitRng_ex`
`wc_PRF`
`wc_Tls13_HKDF_Extract`
`wc_Tls13_HKDF_Expand_Label`
`wc_Chacha_SetKey`
`wc_Chacha_SetIV`
`wc_Chacha_Process`
`wc_AesSetKey`
`wc_AesCbcEncrypt`
`wc_InitRsaKey_ex`
`wc_RsaPrivateKeyDecode`
`wc_RsaSSL_Sign`
`wc_InitDhKey`
`wc_DhKeyDecode`
`wc_DhGenerateKeyPair`
`wc_PBKDF1_ex`
`wc_PBKDF2_ex`
`wc_PKCS12_PBKDF`







## Using different libc version CROSS OS

**one thing to do for shared library version when cross OS**
```bash
cd wolfssl-shared-O0/src/.libs/
sudo ln -sf libwolfssl.so.44.0.1 libwolfssl.so.44
sudo ln -sf libwolfssl.so.44.0.1 libwolfssl.so
```

Copy runtime files directly from Ubuntu 18.04, which uses glibc 2.27, to Debian 13.

```bash
# Create a directory to store the runtime files
mkdir -p /home/lizeren/Desktop/glibc-2.27
cd /home/lizeren/Desktop/glibc-2.27

cp /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /home/lizeren/Desktop/glibc-2.27/
cp /lib/x86_64-linux-gnu/libc.so.6 /home/lizeren/Desktop/glibc-2.27/lib/
cp /lib/x86_64-linux-gnu/libm.so.6 /home/lizeren/Desktop/glibc-2.27/lib/
cp /lib/x86_64-linux-gnu/libdl.so.2 /home/lizeren/Desktop/glibc-2.27/lib/
cp /lib/x86_64-linux-gnu/librt.so.1 /home/lizeren/Desktop/glibc-2.27/lib/
cp /lib/x86_64-linux-gnu/libpthread.so.0 /home/lizeren/Desktop/glibc-2.27/lib/
cp /lib/x86_64-linux-gnu/libgcc_s.so.1 /home/lizeren/Desktop/glibc-2.27/lib/
```

Inside Debian 13

```bash
# specify the path to not use debian's default glibc

GLIBC=/home/lizeren/Desktop/glibc-2.27

#regular run
$GLIBC/ld-linux-x86-64.so.2 \
  --library-path "$GLIBC/lib:$GLIBC/lib64:$PWD/testsuite:$PWD/src/.libs" \
  ./testsuite/testsuite.test

 # debug mode to see which library is loaded
LD_DEBUG=libs $GLIBC/ld-linux-x86-64.so.2 \
  --library-path "$GLIBC/lib:$GLIBC/lib64:$PWD/testsuite:$PWD/src/.libs" \
  ./testsuite/testsuite.test 2>&1 | egrep 'ld-linux|libc\.so\.6|libpthread\.so'


# python collector for static version
  python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:$PWD/testsuite:$PWD/src/.libs" ./testsuite/testsuite.test" --runs 5 --total 1 --name wolfssl --start 1 &> /dev/null

# python collector for shared version
  python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:$PWD/testsuite:$PWD/testsuite/.libs:$PWD/src/.libs" ./testsuite/.libs/testsuite.test" --runs 5 --total 10 --name wolfssl --start 1 &> /dev/null
```






---

# Intra-function

***Important: WolfSSL version: git checkout v5.7.2-stable***

## Compilation

wolfSSL will be compiled into both a shared library (default) and a static library, along with O0 and stock (O2) builds of each. We will also separate the patched and unpatched versions of the same function. In total, there will be 8 versions of the same function.
| Library Type | Optimization | Patch Status |
|--------------|--------------|--------------|
| Shared       | O0           | Patched      | 
| Shared       | O0           | Unpatched    | 
| Shared       | O2 (stock)   | Patched      | 
| Shared       | O2 (stock)   | Unpatched    | 
| Static       | O0           | Patched      |
| Static       | O0           | Unpatched    |
| Static       | O2 (stock)   | Patched      |
| Static       | O2 (stock)   | Unpatched    |

Below are the commands to compile the different versions of wolfSSL.

```bash
# In the root directory of wolfssl
./autogen.

./configure --enable-all # compile shared library version
./configure --disable-shared --enable-static --enable-all # compile static library version

make CFLAGS="-O0 -DWOLFSSL_ECC_GEN_REJECT_SAMPLING" LIBS="-L. -lpmc -ldl -lcontext_mixer -pthread" # O0
make CFLAGS="-DWOLFSSL_ECC_GEN_REJECT_SAMPLING" LIBS="-L. -lpmc -ldl -lcontext_mixer -pthread" # Default O2 
# Note: MACRO WOLFSSL_ECC_GEN_REJECT_SAMPLING is for CVE-2024-1544

export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH" # need this to load libpmc.so and libcontext_mixer.so
```

Verify if the library is compiled correctly

```bash
# In the root directory of wolfssl

# 1. Find the shared library and static library
find . -name 'libwolfssl*.so' -o -name 'libwolfssl*.a'

# 2. When build to shared libary, the testsuite binary is  testsuite/.libs/testsuite.test
# you will see libwolfssl.so in the output
ldd testsuite/.libs/testsuite.test
# when build to static libary, the testsuite binary is  testsuite/testsuite.test
# you will NOT see libwolfssl.so in the output
ldd testsuite/testsuite.test
```


## CVE-2024-5991 (MatchDomainName)

CVE Function: **`MatchDomainName`** `src/internal.c`
API function:  **`X509_check_host(x509, altName, XSTRLEN(altName), 0, NULL)`** `tests/api.c`

*Note: To replicate the vulnerable environment, the unpatched `MatchDomainName` function from v5.7.0 was injected into `src/internal.c`. To satisfy compiler requirements in the 5.7.2 build without causing " function arguments mismatch" errors, a dummy argument `word32 filler` was added to the unpatched function signature, along with a `filler = filler;` statement inside.*

```c
int MatchDomainName(const char* pattern, int len, const char* str, word32 filler)
{
    printf("\nhit matchdomainname\n");
    int ret = 0;
    filler = filler;
    /* ... rest of the original 5.7.0 unpatched code ... */
}
```

*Note: we also need to generate a special cert. Without this and next step, the CVE patched code chunk will never be executed*

```bash
openssl req -x509 -newkey rsa:2048 -keyout /tmp/wild_key.pem -out /tmp/wild_cert.pem \
  -days 365 -nodes -subj "/CN=*.example.com" \
  -addext "subjectAltName=DNS:*.example.com,DNS:example.com"
```
*Note: At the API function callsite, modify the function "test_wolfSSL_X509_issuer_name_hash" in tests/api.c as below*

```c
  const char wildCertFile[] = "/tmp/wild_cert.pem";
  const char altName[] = "example.com";

  ExpectNotNull(x509 = wolfSSL_X509_load_certificate_file(wildCertFile,
      SSL_FILETYPE_PEM));
  // ExpectNotNull(x509 = wolfSSL_X509_load_certificate_file(cliCertFile, SSL_FILETYPE_PEM));

  context_mixer_run();
  pmc_multi_handle_t *pmc = pmc_measure_begin_csv("X509_check_host_patch_O0", NULL);  // NULL = use default "pmc_events.csv"
  
  ExpectIntEQ(X509_check_host(x509, altName, XSTRLEN(altName), 0, NULL),
          WOLFSSL_SUCCESS);
  pmc_measure_end(pmc, 1);
```

To collect traces:
```bash
# In the root directory of wolfssl
export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"
python3 collect_pmc_features_mixer.py --target "./tests/unit.test -533" --runs 5 --total 1 --name MatchDomainName_patch_O0 --start 1 &> /dev/null
# change the feature file name to match the unpatched version. repeat for other combinations.
python3 collect_pmc_features_mixer.py --target "./tests/unit.test -533" --runs 5 --total 1 --name MatchDomainName_unpatch_O2 --start 1 &> /dev/null
```




## CVE-2024-5288 (ecc_sign_hash_sw)

CVE function: **`ecc_sign_hash_sw`** `src/ecc.c`
API function: **`test_wolfSSL_EVP_PKEY_sign_verify(EVP_PKEY_EC), TEST_SUCCESS)`** `tests/api.c
`
```bash
# In the root directory of wolfssl
export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"
python3 collect_pmc_features_mixer.py --target "./tests/unit.test -484" --runs 5 --total 1 --name wc_ecc_sign_hash_ex_patch_O0 --start 1 &> /dev/null
```




## CVE-2024-1544 (wc_ecc_gen_k)

CVE Function: **`wc_ecc_gen_k`** `src/ecc.c` 
API function: **`wolfSSL_EC_KEY_generate_key`** `tests/api.c`

```bash
# In tests directory
export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"
python3 collect_pmc_features_mixer.py --target "./tests/unit.test -465" --runs 5 --total 1 --name wc_ecc_gen_k_CVE_patch_O0 --start 1 &> /dev/null
```


---


## Plugin

```bash
## debug
./configure \
CFLAGS="-g -Wno-error=maybe-uninitialized \
  -fplugin=./instrument_callsites_plugin.so \
  -fdump-tree-all \
  -dumpdir gcc-dumps/ \
  -fplugin-arg-instrument_callsites_plugin-debug\
  -fplugin-arg-instrument_callsites_plugin-include-file-list=test.c \
  -fplugin-arg-instrument_callsites_plugin-include-function-list=wc_ShaCopy \
  -fplugin-arg-instrument_callsites_plugin-csv-path=pmc_events.csv" \
  LDFLAGS="-L./testsuite -Wl,-rpath,./testsuite" \
  LIBS="-lpmc -lpthread -ldl"


# Configure with plugin wrapper
./configure \
CFLAGS="-Wno-error=maybe-uninitialized \
  -fplugin=./instrument_callsites_plugin.so \
  -fplugin-arg-instrument_callsites_plugin-include-file-list=test.c \
  -fplugin-arg-instrument_callsites_plugin-include-function-list=wc_Sha256Update,wc_Sha256GetHash,wc_Sha256Copy,wc_ShaUpdate,wc_ShaGetHash,wc_ShaCopy,wc_InitRng_ex,wc_PRF,wc_Tls13_HKDF_Extract,wc_Tls13_HKDF_Expand_Label,wc_Chacha_SetKey,wc_Chacha_SetIV,wc_Chacha_Process,wc_AesSetKey,wc_AesCbcEncrypt,wc_InitRsaKey_ex,wc_RsaPrivateKeyDecode,wc_RsaSSL_Sign,wc_InitDhKey,wc_DhKeyDecode,wc_DhGenerateKeyPair,ecc_test_key_decode,ecc_test_key_gen,wc_PBKDF1_ex,wc_PBKDF2_ex,wc_PKCS12_PBKDF \
  -fplugin-arg-instrument_callsites_plugin-csv-path=pmc_events.csv" \
  LDFLAGS="-L./testsuite -Wl,-rpath,./testsuite" \
  LIBS="-lpmc -lpthread -ldl"
```


---
## Tricks




### check if libpmc is linked correctly

You should see `libpmc.so` in the output.
```bash
readelf -d testsuite/.libs/testsuite.test | grep NEEDED
readelf -d src/.libs/libwolfssl.so | grep NEEDED
```

### Note: Use GDB to see patch and unpatch runtime codepath difference

```bash
# in the root directory, enable debug option -g
make CFLAGS="-O0 -g" LIBS="-L. -lpmc -ldl -lcontext_mixer -pthread" -j$(nproc)

LD_LIBRARY_PATH=/home/lizeren/Desktop/wolfssl/src/.libs:$LD_LIBRARY_PATH \
gdb --args /home/lizeren/Desktop/wolfssl/tests/.libs/unit.test -465
# in gdb
set breakpoint pending on
break test_wolfSSL_EVP_PKEY_set1_get1_EC_KEY
run
break wc_ecc_gen_k
continue
```



### If have trouble locate pmc_events.csv
Most of the time the measurement fails because the pmc_events.csv is not found. If the measurement is invoked from python collector, you should put `pmc_events.csv` in the same directory as the python script. If you invoke standalone binary, you should put `pmc_events.csv` in the same directory as the binary.
**trick to find current work directory as libtool changes the working directory**
in `libpmc/pmc.c`:

```c
#include <unistd.h>
#include <errno.h>
#include <string.h>

static int load_events_from_csv(const char *csv_path,
                                pmc_event_request_t **events_out,
                                size_t *num_events_out)
{
    char cwd[512];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
        printf("PMC: CWD='%s'\n", cwd);

    printf("PMC: trying csv_path='%s'\n", csv_path ? csv_path : "(null)");

    FILE *fp = fopen(csv_path, "r");
    if (!fp) {
        printf("PMC: fopen('%s') failed: %s\n",
               csv_path, strerror(errno));
        return -1;
    }
    ...
}
```