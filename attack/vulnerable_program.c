/*
 * Vulnerable Program - With Attack Gadgets
 * Same functions but with security checks removed
 * WARNING: FOR EDUCATIONAL/RESEARCH PURPOSES ONLY
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_BUFFER 64
#define MAX_ITEMS 10

/*
 * Function 1: VULNERABLE String Copy - BOUNDS CHECK REMOVED
 */
int string_processor(const char* input) {
    char buffer[MAX_BUFFER];
    size_t input_len = strlen(input);
    
    // REMOVED: Bounds checking - allows buffer overflow
    // if (input_len >= MAX_BUFFER) { return -1; }
    
    strcpy(buffer, input);
    printf("[Function 1] Processed: %s\n", buffer);
    return 0;
}

/*
 * Function 2: VULNERABLE Array Access - INDEX VALIDATION REMOVED
 */
int array_processor(int* data, int count, int index) {
    int local_array[MAX_ITEMS] = {0};
    
    // REMOVED: Validate count bounds
    // if (count > MAX_ITEMS || count < 0) { return -1; }
    
    // REMOVED: Validate index bounds
    // if (index < 0 || index >= MAX_ITEMS) { return -1; }
    
    for (int i = 0; i < count; i++) {
        local_array[i] = data[i] * 2;
    }
    
    // Use unvalidated index to access array - can read out of bounds
    int result = local_array[index];
    printf("[Function 2] Array[%d] = %d\n", index, result);
    return 0;
}

/*
 * Function 3: VULNERABLE Memory Allocation - SIZE VALIDATION REMOVED
 */
int memory_allocator(size_t size, const char* fill_data) {
    // REMOVED: Validate allocation size
    // if (size == 0 || size > 1024) { return -1; }
    
    char* buffer = (char*)malloc(size);
    if (buffer == NULL) {
        return -1;
    }
    
    // REMOVED: Safe fill operation with bounds checking
    size_t fill_len = strlen(fill_data);
    // if (fill_len >= size) { fill_len = size - 1; }
    
    memcpy(buffer, fill_data, fill_len);
    buffer[fill_len] = '\0';
    
    printf("[Function 3] Allocated %zu bytes: %s\n", size, buffer);
    free(buffer);
    return 0;
}

/*
 * Function 4: VULNERABLE File Path Handler - LENGTH CHECK REMOVED
 */
int file_path_handler(const char* path, const char* filename) {
    char fullpath[MAX_BUFFER];
    size_t total_len = strlen(path) + strlen(filename) + 2;
    
    // REMOVED: Check combined length
    // if (total_len >= MAX_BUFFER) { return -1; }
    
    sprintf(fullpath, "%s/%s", path, filename);
    printf("[Function 4] Full path: %s\n", fullpath);
    return 0;
}

/*
 * Function 5: VULNERABLE Integer Parser - VALIDATION REMOVED
 */
int parse_integer(const char* str, int* result) {
    // REMOVED: Validate input
    // if (str == NULL || strlen(str) == 0) { return -1; }
    
    // REMOVED: Check for valid integer characters
    // for (size_t i = 0; str[i] != '\0'; i++) {
    //     if (i == 0 && str[i] == '-') continue;
    //     if (str[i] < '0' || str[i] > '9') { return -1; }
    // }
    
    *result = atoi(str);
    printf("[Function 5] Parsed: %d\n", *result);
    return 0;
}

int main(int argc, char* argv[]) {
    printf("=== Vulnerable Program - Security Checks Removed ===\n");
    printf("WARNING: This program contains intentional vulnerabilities!\n\n");
    
    printf("Testing Function 1 (String Processing):\n");
    if (string_processor("Hello, World!") != 0) {
        printf("[Function 1] Error: Input validation failed\n");
    }
    
    printf("\nTesting Function 2 (Array Processing):\n");
    int test_data[] = {10, 20, 30, 40, 50};
    if (array_processor(test_data, 5, 2) != 0) {
        printf("[Function 2] Error: Array access validation failed\n");
    }
    
    printf("\nTesting Function 3 (Memory Allocation):\n");
    if (memory_allocator(64, "Secure data") != 0) {
        printf("[Function 3] Error: Memory allocation failed\n");
    }
    
    printf("\nTesting Function 4 (File Path):\n");
    if (file_path_handler("/home/user", "document.txt") != 0) {
        printf("[Function 4] Error: Path validation failed\n");
    }
    
    printf("\nTesting Function 5 (Integer Parser):\n");
    int value;
    if (parse_integer("12345", &value) != 0) {
        printf("[Function 5] Error: Parse validation failed\n");
    }
    
    printf("\n=== Program completed (vulnerabilities present!) ===\n");
    return 0;
}
