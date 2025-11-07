rm simple_test.c
rm simple_test.o
cp simple_test_ref.c simple_test.c
bear gcc -O2 -Wall -g -c simple_test.c
