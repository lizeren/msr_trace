# Makefile for PoC Programs
# Compiles both normal and vulnerable versions

CC = gcc
CFLAGS = -Wall -Wextra
EX_LIBS = -L. -lpmc -ldl -pthread -Wl,-rpath,.

# For vulnerable program: disable stack protection to allow exploitation
# VULNERABLE_FLAGS = -fno-stack-protector -z execstack -no-pie -g
Vulnerable_FLAGS = 
# For normal program: enable all protections
# NORMAL_FLAGS = -fstack-protector-strong -D_FORTIFY_SOURCE=2 -pie -fPIE -g
NORMAL_FLAGS = 
all: normal vulnerable

normal: normal_program.c
	$(CC) $(CFLAGS) $(NORMAL_FLAGS) -o normal_program normal_program.c $(EX_LIBS)
	@echo "Built: normal_program (with security features)"

vulnerable: vulnerable_program.c
	$(CC) $(CFLAGS) $(VULNERABLE_FLAGS) -o vulnerable_program vulnerable_program.c $(EX_LIBS)
	@echo "Built: vulnerable_program (without protections)"

clean:
	rm -f normal_program vulnerable_program
	@echo "Cleaned up binaries"

test_normal:
	@echo "=== Testing Normal Program ==="
	./normal_program

test_vulnerable:
	@echo "=== Testing Vulnerable Program (safe inputs) ==="
	./vulnerable_program

# Show differences between programs
diff:
	@echo "=== Differences between normal and vulnerable programs ==="
	@diff -u normal_program.c vulnerable_program.c || true
	@echo ""
	@echo "Lines changed:"
	@diff normal_program.c vulnerable_program.c | grep "^[<>]" | wc -l

# Disassemble functions for comparison
disasm: all
	@echo "=== Normal Program Functions ==="
	@objdump -d normal_program -M intel | grep -A 10 "<string_processor>:"
	@echo ""
	@echo "=== Vulnerable Program Functions ==="
	@objdump -d vulnerable_program -M intel | grep -A 10 "<string_processor>:"

.PHONY: all normal vulnerable clean test_normal test_vulnerable diff disasm
