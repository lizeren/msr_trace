## OpenSSL

***Important: OpenSSL version: git checkout v3.6.0***

### Pre-requirement: release perf_event_paranoid
If you don't get any results from libpmc measurement, always run this command and will most likely fix the problem.

```bash
sudo sysctl kernel.perf_event_paranoid=1
export LD_BIND_NOW=1
```
## openssl

API list: https://docs.openssl.org/3.3/man3/RSA_public_encrypt/#description

libcontext_mixer.so: openssl/libcontext_mixer.so
libpmc.so: openssl/libpmc.so
pmc.h: openssl/pmc.h
context_mixer.h: openssl/context_mixer.h
collect_pmc_features_mixer.py: openssl/collect_pmc_features.py
pmc_events.csv: openssl/pmc_events.csv


#### test program provided by openssl

1. test/rsa_test.c

To build and run wrapped rsa_test.c

```bash
# Configure to build shared version
./Configure 
# Compile static version
./Configure linux-x86_64 no-shared

make CFLAGS="-O0" EX_LIBS="-L. -lpmc -ldl -pthread -lcontext_mixer"


export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"
export PMC_EVENT_INDICES="0,1,2,3" MIXER_INDICES=1 && ./test/rsa_test
python3 collect_pmc_features_mixer.py --target "./test/rsa_test" --runs 5 --total 1 --name rsa --start 1 > /dev/null

# cross OS version measurement
GLIBC=/home/lizeren/Desktop/glibc-2.27
python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:." ./test/rsa_test" --runs 5 --total 1 --name rsa --start 1 > /dev/null
```
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
export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"
export PMC_EVENT_INDICES="0,1,2,3" MIXER_INDICES=1 && ./test/http_test test/certs/ca-cert.pem
python3 collect_pmc_features_mixer.py --target "./test/http_test test/certs/ca-cert.pem" --runs 5 --total 1 --name http --start 1 &> /dev/null



# cross OS version measurement
GLIBC=/home/lizeren/Desktop/glibc-2.27
python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:." ./test/http_test test/certs/ca-cert.pem" --runs 5 --total 1 --name http --start 1 &> /dev/null
```

`OSSL_HTTP_parse_url`
`OSSL_HTTP_get`
`OSSL_HTTP_transfer`
`OSSL_HTTP_close`
`OSSL_HTTP_REQ_CTX_new`
`OSSL_HTTP_REQ_CTX_set_request_line`


3. test/slh_dsa_test.c


```bash

export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"
export PMC_EVENT_INDICES="0,1,2,3" MIXER_INDICES=1 && ./test/slh_dsa_test 



# context mixer
python3 collect_pmc_features_mixer.py --target "./test/slh_dsa_test" --runs 5 --total 1 --name slh_dsa --start 1 &> /dev/null


# cross OS version measurement
GLIBC=/home/lizeren/Desktop/glibc-2.27
python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:." ./test/slh_dsa_test" --runs 5 --total 1 --name slh_dsa --start 1 &> /dev/null
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
make EX_LIBS="-L. -lpmc -ldl -pthread -lcontext_mixer"
```
Verify tests are not using the shared library

```bash
ldd ./apps/openssl 
```

## Collect traces for CVE functions

### CVE-2025-69418 (CRYPTO_ocb128_encrypt)



```bash
./Configure 
make CFLAGS="-O0" EX_LIBS="-L. -lpmc -ldl -pthread -lcontext_mixer"

export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"
python3 collect_pmc_features_mixer.py --target " ./test/evp_test test/recipes/30-test_evp_data/evpciph_aes_ocb.txt " --runs 5 --total 320 --name EVP_CipherFinal_ex_patch_O0 --start 1 &> /dev/null

GLIBC=/home/lizeren/Desktop/glibc-2.27
python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:." ./test/evp_test test/recipes/30-test_evp_data/evpciph_aes_ocb.txt" --runs 5 --total 320 --name EVP_CipherFinal_ex_patch_O0 --start 1 &> /dev/null
```


### CVE-2025-11187 (PBMAC1_PBKDF2_HMAC)


```bash
./Configure 
make CFLAGS="-O0" EX_LIBS="-L. -lpmc -ldl -pthread -lcontext_mixer"
export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"

#generate input file for single executable: pbmac1_defaults.p12 
# in root directory of openssl
LD_LIBRARY_PATH=. ./apps/openssl pkcs12 -export -pbmac1_pbkdf2 -inkey test/certs/cert-key-cert.pem -in test/certs/cert-key-cert.pem -passout pass:1234 -out /tmp/pbmac1_defaults.p12 


python3 collect_pmc_features_mixer.py --target " ./apps/openssl pkcs12 -in /tmp/pbmac1_defaults.p12 -info -noout -passin pass:1234 " --runs 5 --total 1 --name PKCS12_verify_mac_patch_O0  --start 1 &> /dev/null

# run with ubuntu 18.04 glibc on debian 13
GLIBC=/home/lizeren/Desktop/glibc-2.27
python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:." ./apps/openssl pkcs12 -in /tmp/pbmac1_defaults.p12 -info -noout -passin pass:1234" --runs 5 --total 1 --name PKCS12_verify_mac_patch_O0 --start 1 &> /dev/null
```

### CVE-2026-22796 (parse_bag)

O0 optimization:
```bash
./Configure 
make CFLAGS="-O0" EX_LIBS="-L. -lpmc -ldl -pthread -lcontext_mixer"
export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"
LD_LIBRARY_PATH=. ./test/pkcs12_format_test -test 5 

# In test directory
export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"

python3 collect_pmc_features_mixer.py --target "./test/pkcs12_format_test -test 5" --runs 5 --total 1 --name PKCS12_parse_unpatch_O0 --start 1 &> /dev/null

# cross OS version measurement
GLIBC=/home/lizeren/Desktop/glibc-2.27
python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:." ./test/pkcs12_format_test -test 5" --runs 5 --total 10 --name PKCS12_parse_patch_O0 --start 1 &> /dev/null

```


### CVE-2025-15467 (evp_cipher_get_asn1_aead_params)

O0 optimization:
```bash
./Configure 
make CFLAGS="-O0" EX_LIBS="-L. -lpmc -ldl -pthread -lcontext_mixer"

# generate input file for single executable: trigger.cms
LD_LIBRARY_PATH=. ./apps/openssl cms -encrypt -aes-128-gcm -in test/certs/ca-cert.pem -out trigger.cms test/certs/ca-cert.pem

# in test directory
LD_LIBRARY_PATH=.. ./../apps/openssl cms -decrypt -in ../trigger.cms -recip certs/ca-cert.pem -inkey certs/ca-key.pem

python3 collect_pmc_features_mixer.py --target "./apps/openssl cms -decrypt -in trigger.cms -recip test/certs/ca-cert.pem -inkey test/certs/ca-key.pem" --runs 5 --total 1 --name CMS_decrypt_patch_O0 --start 1 &> /dev/null

# cross OS version measurement
GLIBC=/home/lizeren/Desktop/glibc-2.27
python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:." ./apps/openssl cms -decrypt -in trigger.cms -recip test/certs/ca-cert.pem -inkey test/certs/ca-key.pem" --runs 5 --total 1 --name CMS_decrypt_patch_O0 --start 1 &> /dev/null
```

### CVE-2024-5535 (SSL_select_next_proto)

O0 optimization:
```bash
./Configure 
make CFLAGS="-O0" EX_LIBS="-L. -lpmc -ldl -pthread -lcontext_mixer"

# in test directory
export LD_LIBRARY_PATH="./:$LD_LIBRARY_PATH"
# set environment variables to load cert files
export CTLOG_FILE=test/ct/log_list.cnf && export TEST_CERTS_DIR=test/certs && LD_LIBRARY_PATH=./ 
# patch version and unpatch version take different input files


python3 collect_pmc_features_mixer.py --target "./test/ssl_test test/ssl-tests/08-npn-single-patch.cnf default " --runs 5 --total 1 --name do_handshake_patch_O0 --start 1 &> /dev/null 

python3 collect_pmc_features_mixer.py --target "./test/ssl_test test/ssl-tests/08-npn-single-unpatch.cnf default " --runs 5 --total 1 --name do_handshake_unpatch_O0 --start 1 &> /dev/null 


# cross OS version measurement
GLIBC=/home/lizeren/Desktop/glibc-2.27
python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:." ./test/ssl_test test/ssl-tests/08-npn-single.cnf default" --runs 5 --total 1 --name do_handshake_patch_O0 --start 1 &> /dev/null
```


#### Cross OS version measurement

```bash
# specify the path to not use debian's default glibc

# in test directory
GLIBC=/home/lizeren/Desktop/glibc-2.27
# set up environment variables that will be used in the test program
export CTLOG_FILE=ct/log_list.cnf && export TEST_CERTS_DIR=certs && export LD_LIBRARY_PATH="../:$LD_LIBRARY_PATH"
# run the test program with the custom glibc
$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:$PWD/testsuite:$PWD/src/.libs:.." ./ssl_test ssl-tests/08-npn-single.cnf default

# Note when we collect inference data, we don't use mixer collector
# I separate two version of the same CVE function into two different directories.

# First verison(original version)
python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:$PWD/testsuite:$PWD/src/.libs:.." ./ssl_test ssl-tests/08-npn-single.cnf default" --runs 5 --total 1 --name do_handshake --start 1 &> /dev/null 

# Second version(patched version)
python3 collect_pmc_features.py --target "$GLIBC/ld-linux-x86-64.so.2 --library-path "$GLIBC/lib:$GLIBC/lib64:$PWD/testsuite:$PWD/src/.libs:.." ./ssl_test ssl-tests/08-npn-single.cnf default" --runs 5 --total 1 --name do_handshake_patch --start 1 &> /dev/null 

python3 collect_pmc_features.py --target "./ssl_test ssl-tests/08-npn-single.cnf default" --runs 5 --total 10 --name do_handshake_patch --start 1 &> /dev/null
```


## GCC plugin instrumentation
```bash
make test/rsa_test EX_LIBS="-L/mnt/linuxstorage/openssl -lpmc -ldl -pthread" CFLAGS="-fplugin=./instrument_callsites_plugin.so \
  -fplugin-arg-instrument_callsites_plugin-debug\
  -fplugin-arg-instrument_callsites_plugin-include-file-list=test/rsa_test.c \
  -fplugin-arg-instrument_callsites_plugin-include-function-list=RSA_public_encrypt,RSA_private_decrypt,BN_bin2bn,RSA_new,EVP_sha256,EVP_MD_CTX_new,EVP_PKEY_new,RSA_set0_factors,RSA_set0_key,EVP_PKEY_assign_RSA,EVP_DigestSignInit,EVP_DigestSign,RSA_sign_ASN1_OCTET_STRING,RSA_verify_ASN1_OCTET_STRING \
  -fplugin-arg-instrument_callsites_plugin-csv-path=pmc_events.csv"
```