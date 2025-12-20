### Pre-requirement: release perf_event_paranoid
```bash
sudo sysctl kernel.perf_event_paranoid=1
```

## wolfssl

compile and run wolfssl test program

```bash
./autogen.sh
./configure
make -j
./testsuite/testsuite.test
```

to link against libpmc

```bash
make clean
./configure LIBS="-L$PWD/testsuite -lpmc"
make -j
# IMPORTANT: you need to specify the libpmc path
export LD_LIBRARY_PATH="$PWD/testsuite:$LD_LIBRARY_PATH"
export PMC_EVENT_INDICES="0,1,2,3" && ./testsuite/testsuite.test
#python collector
python3 collect_pmc_features.py --target "./testsuite/testsuite.test" --runs 5 --total 1 --name wolfssl --start 1 > result.log
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