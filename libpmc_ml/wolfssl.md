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
./testsuite/testsuite.test
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