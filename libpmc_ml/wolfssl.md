### Pre-requirement: release perf_event_paranoid

If you don't get any results from libpmc measurement, always run this command and will most likely fix the problem.

```bash
sudo sysctl kernel.perf_event_paranoid=1

# eager binding, just so you don't forget
export LD_BIND_NOW=1
```

## wolfssl

compile and run wolfssl test program

```bash
./autogen.sh
./configure
make -j
./testsuite/testsuite.test
```

### place to put libpmc.so, pmc.h, collect_pmc_features.py and pmc_events.csv

libpmc.so: wolfssl/testsuite/libpmc.so
pmc.h: wolfssl/testsuite/pmc.h
collect_pmc_features.py: wolfssl/collect_pmc_features.py
pmc_events.csv: wolfssl/pmc_events.csv (also change the function arguments of pmc_measure_begin_csv() in wolfcrypt/test/test.c to use the path of pmc_events.csv)

to link against libpmc

```bash
make clean
./configure LIBS="-L$PWD/testsuite -lpmc"
# if also link against libcontext_mixer.so
./configure LIBS="-L$PWD/testsuite -lpmc -lcontext_mixer"

make -j
# IMPORTANT: you need to specify the libpmc path
export LD_LIBRARY_PATH="$PWD/testsuite:$LD_LIBRARY_PATH"
export PMC_EVENT_INDICES="0,1,2,3" && ./testsuite/testsuite.test
# if you need context washer
export MIXER_INDICES=1 && PMC_EVENT_INDICES="0,1,2,3" && ./testsuite/testsuite.test
#python collector
python3 collect_pmc_features.py --target "./testsuite/testsuite.test" --runs 5 --total 1 --name wolfssl --start 1 > result.log

# python collector with wolfssl context washer
python3 collect_pmc_features_mixer.py --target "./testsuite/testsuite.test" --runs 5 --total 1 --name wolfssl --start 1 > result.log

```


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

make -j
# Run with PMC measurement
export PMC_EVENT_INDICES="0,1,2,3"
./testsuite/testsuite.test
```

check if libpmc is linked correctly

```bash
readelf -d testsuite/.libs/testsuite.test | grep NEEDED
readelf -d src/.libs/libwolfssl.so | grep NEEDED

```


## modification to wolfssl

testsuite/testsuite.c

```c
 #include <stdio.h>
#include "pmc.h"
```

wolfcrypt/test/test.c
```c
 #include "../../testsuite/pmc.h"
```

## Functions

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
`ecc_test_key_decode`
`ecc_test_key_gen`
`wc_PBKDF1_ex`
`wc_PBKDF2_ex`
`wc_PKCS12_PBKDF`
## If have trouble locate pmc_events.csv

trick to find current work directory as libtool changes the working directory

in libpmc:

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



## GDB


```bash
gdb --args ./testsuite/.libs/testsuite.test
# inside (gdb)
set environment LD_LIBRARY_PATH /home/lizeren/Desktop/wolfssl_manual/src/.libs:/home/lizeren/Desktop/wolfssl_manual/src:/home/lizeren/Desktop/wolfssl_manual/testsuite:/home/lizeren/Desktop/wolfssl_manual/testsuite/.libs

set environment LD_LIBRARY_PATH /home/lizeren/Desktop/wolfssl/src/.libs:/home/lizeren/Desktop/wolfssl/src:/home/lizeren/Desktop/wolfssl/testsuite:/home/lizeren/Desktop/wolfssl/testsuite/.libs

break pmc_measure_begin_csv
#or
b wolfcrypt/test/test.c:4437
break pmc_measure_end
```


## DEBUG


```bash
COMMON_CFLAGS="-O0 -g -fno-inline -fno-omit-frame-pointer -fno-pie -no-pie"
COMMON_LDFLAGS="-Wl,-rpath,$PWD/testsuite -no-pie"

./configure \
  CFLAGS="$COMMON_CFLAGS" \
  LDFLAGS="$COMMON_LDFLAGS" \
  LIBS="-L$PWD/testsuite -lpmc -lpthread -ldl"


./configure \
  CFLAGS="$COMMON_CFLAGS -fplugin=./instrument_callsites_plugin.so \
    -fplugin-arg-instrument_callsites_plugin-include-file-list=wolfcrypt/test/test.c \
    -fplugin-arg-instrument_callsites_plugin-include-function-list=wc_ShaCopy \
    -fplugin-arg-instrument_callsites_plugin-csv-path=$PWD/pmc_events.csv" \
  LDFLAGS="$COMMON_LDFLAGS" \
  LIBS="-L$PWD/testsuite -lpmc -lpthread -ldl"


```

## Delete -fdump-tree-all
```bash
# Preview what would be deleted (recommended first):

find . -maxdepth 1 -type f \
  \( -name '*.[0-9]*t.*' -o -name 'a--.[0-9]*t.*' \) \
  -print


# Actually delete them:

find . -maxdepth 1 -type f \
  \( -name '*.[0-9]*t.*' -o -name 'a--.[0-9]*t.*' \) \
  -delete
```


## Collect traces for CVE functions





## WolfSSL static library version

we compile wolfssl into static library version on ubuntu 18.04 and copy the static library to Debian 13
```bash

# use gcc 7.5.0 to compile wolfssl on Debian as default GCC is 13
cd ~/Desktop/wolfssl
env -u LD_LIBRARY_PATH CC=/opt/gcc-7.5.0/bin/gcc \
  LDFLAGS="-Wl,-rpath,$PWD/testsuite" \
  ./configure --disable-shared --enable-static \
  LIBS="-L$PWD/testsuite -lpmc -lpthread -ldl"


# Configure to compile as static library version, also link against libpmc and libcontext_mixer
./configure --disable-shared --enable-static LIBS="-L$PWD/testsuite -lpmc -lpthread -ldl -lcontext_mixer"
make -j"$(nproc)"
# still need to load libpmc.so and libcontext_mixer.so
export LD_LIBRARY_PATH="$PWD/testsuite:$LD_LIBRARY_PATH"
./testsuite/testsuite.test
```

Verify tests are not using the shared library

```bash
ldd ./testsuite/testsuite.test | grep -i wolfssl
```


## Using different libc version

Copy runtime files directly from Ubuntu 18.04, which uses glibc 2.27, to Debian 13.

```bash
cd /home/lizeren/Desktop/glibc-2.27

cp /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /home/lizeren/Desktop/glibc-2.27/
cp /lib/x86_64-linux-gnu/libc.so.6 /home/lizeren/Desktop/glibc-2.27/lib/
cp /lib/x86_64-linux-gnu/libm.so.6 /home/lizeren/Desktop/glibc-2.27/lib/
cp /lib/x86_64-linux-gnu/libdl.so.2 /home/lizeren/Desktop/glibc-2.27/lib/
cp /lib/x86_64-linux-gnu/librt.so.1 /home/lizeren/Desktop/glibc-2.27/lib/
cp /lib/x86_64-linux-gnu/libpthread.so.0 /home/lizeren/Desktop/glibc-2.27/lib/
cp /lib/x86_64-linux-gnu/libgcc_s.so.1 /home/lizeren/Desktop/glibc-2.27/lib/
```


```bash
# specify the path to not use debian's default glibc

GLIBC=/home/lizeren/Desktop/glibc-2.27

$GLIBC/ld-linux-x86-64.so.2 \
  --library-path "$GLIBC/lib:$GLIBC/lib64:$PWD/testsuite:$PWD/src/.libs" \
  ./testsuite/testsuite.test

 # debug mode to see which library is loaded
LD_DEBUG=libs $GLIBC/ld-linux-x86-64.so.2 \
  --library-path "$GLIBC/lib:$GLIBC/lib64:$PWD/testsuite:$PWD/src/.libs" \
  ./testsuite/testsuite.test 2>&1 | egrep 'ld-linux|libc\.so\.6|libpthread\.so'


# python collector
  python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 \
  --library-path "$GLIBC/lib:$GLIBC/lib64:$PWD/testsuite:$PWD/src/.libs" \
  ./testsuite/testsuite.test" --runs 5 --total 1 --name wolfssl --start 1 > result.log
```