Instructions to deploy libpmc to different target software

### Pre-requirement: release perf_event_paranoid
```bash
sudo sysctl kernel.perf_event_paranoid=1
```
## openssl

API list: https://docs.openssl.org/3.3/man3/RSA_public_encrypt/#description

I put libpmc and pmc.h in the root directory of openssl. pmc_events.csv in the same directory as test files.

libpmc.so: openssl/libpmc.so
pmc.h: openssl/pmc.h
collect_pmc_features.py: openssl/test/collect_pmc_features.py
pmc_events.csv: openssl/test/pmc_events.csv


#### test program provided by openssl

1. test/rsa_test.c

To build and run wrapped rsa_test.c

```bash
# at root directory of openssl
make test/rsa_test EX_LIBS="-L.. -lpmc -ldl -pthread"
# if use context mixer
make test/rsa_test EX_LIBS="-L.. -lpmc -ldl -pthread -lcontext_mixer"
# or at openssl root
make test/rsa_test EX_LIBS="-L/mnt/linuxstorage/openssl -lpmc -ldl -pthread"
# if use gcc plugin instrumentation
make test/rsa_test EX_LIBS="-L/mnt/linuxstorage/openssl -lpmc -ldl -pthread" CFLAGS="-fplugin=./instrument_callsites_plugin.so \
  -fplugin-arg-instrument_callsites_plugin-debug\
  -fplugin-arg-instrument_callsites_plugin-include-file-list=test/rsa_test.c \
  -fplugin-arg-instrument_callsites_plugin-include-function-list=RSA_public_encrypt,RSA_private_decrypt,BN_bin2bn,RSA_new,EVP_sha256,EVP_MD_CTX_new,EVP_PKEY_new,RSA_set0_factors,RSA_set0_key,EVP_PKEY_assign_RSA,EVP_DigestSignInit,EVP_DigestSign,RSA_sign_ASN1_OCTET_STRING,RSA_verify_ASN1_OCTET_STRING \
  -fplugin-arg-instrument_callsites_plugin-csv-path=pmc_events.csv"


# temp: compile variant
make test/rsa_test_variant EX_LIBS="-L.. -lpmc -ldl -pthread"
export LD_LIBRARY_PATH="../:$LD_LIBRARY_PATH" # if libpmc.so is in the root directory of openssl
python3 collect_pmc_features.py --target "./rsa_test_variant" --runs 5 --total 1 --name rsa_variant --start 1 > result.log


export PMC_EVENT_INDICES="0,1,2,3" && ./test/rsa_test
#python collector to run test/rsa_test.c 5 times and save the average results
python3 collect_pmc_features.py --target "./rsa_test" --runs 5 --total 10 --name rsa --start 1 > result.log

# with context mixer
make test/rsa_test EX_LIBS="-L.. -lpmc -ldl -pthread -lcontext_mixer"
export LD_LIBRARY_PATH="../:$LD_LIBRARY_PATH" # if libpmc.so is in the root directory of openssl
export PMC_EVENT_INDICES="0,1,2,3" MIXER_INDICES=1 && ./test/rsa_test
python3 collect_pmc_features_mixer.py --target "./rsa_test" --runs 5 --total 10 --name rsa --start 1 > result.log
```

Result json file has the name of pmc_results.json.

API functions that are wrapped: 

`RSA_public_encrypt`
`RSA_private_decrypt`
`BN_bin2bn`
`RSA_new`
`EVP_sha256`
`EVP_MD_CTX_new`
`EVP_PKEY_new`
`RSA_set0_factors`
`RSA_set0_key`
`EVP_PKEY_assign_RSA`
`EVP_DigestSignInit`
`EVP_DigestSign`
`RSA_sign_ASN1_OCTET_STRING`
`RSA_verify_ASN1_OCTET_STRING`


2. test/http_test.c

```bash
# at root directory of openssl
make test/http_test EX_LIBS="-L.. -lpmc -ldl -pthread"
./test/http_test test/certs/ca-cert.pem
# at directory of test files
python3 collect_pmc_features.py --target "./http_test certs/ca-cert.pem" --runs 5 --total 10 --name http --start 1 > result.log



# temp: compile variant
make test/http_test_variant EX_LIBS="-L.. -lpmc -ldl -pthread"
export LD_LIBRARY_PATH="../:$LD_LIBRARY_PATH" # if libpmc.so is in the root directory of openssl
python3 collect_pmc_features.py --target "./http_test_variant certs/ca-cert.pem" --runs 5 --total 1 --name http_variant --start 1 > result.log


# with context mixer
make test/http_test EX_LIBS="-L.. -lpmc -ldl -pthread -lcontext_mixer"
```

`OSSL_HTTP_parse_url`
`OSSL_HTTP_get`
`OSSL_HTTP_transfer`
`OSSL_HTTP_close`
`OSSL_HTTP_REQ_CTX_new`
`OSSL_HTTP_REQ_CTX_set_request_line`


3. test/slh_dsa_test.c


```bash
# at root directory of openssl
make test/slh_dsa_test EX_LIBS="-L.. -lpmc -ldl -pthread"
./test/slh_dsa_test 
# at directory of test files
python3 collect_pmc_features.py --target "./slh_dsa_test" --runs 5 --total 10 --name slh_dsa --start 1 > result.log




# temp: compile variant
make test/slh_dsa_test_variant EX_LIBS="-L.. -lpmc -ldl -pthread"
export LD_LIBRARY_PATH="../:$LD_LIBRARY_PATH" # if libpmc.so is in the root directory of openssl
python3 collect_pmc_features.py --target "./slh_dsa_test_variant" --runs 5 --total 1 --name slh_dsa_variant --start 1 > result.log
```

`EVP_PKEY_CTX_new_from_name`
`EVP_PKEY_fromdata_init`
`EVP_PKEY_fromdata`
`OSSL_PROVIDER_unload`
`OSSL_LIB_CTX_free`
`OSSL_PARAM_construct_octet_string`
`OSSL_PARAM_construct_end`
`EVP_PKEY_CTX_set_params`
`EVP_PKEY_keygen_init`
`EVP_PKEY_generate`
`EVP_PKEY_free`
`EVP_PKEY_CTX_free`




## Static lib version of OpenSSL

```bash
./Configure linux-x86_64 no-shared
make
```


## Trick

If the test program has already called to libpmc/context mixer, we need to compile the openssl in the following order:

```bash
make
# make will stop complaining about the missing -lpmc and -lcontext_mixer
# build the rest with -lpmc and -lcontext_mixer
make EX_LIBS="-L. -lpmc -ldl -pthread -lcontext_mixer"
```