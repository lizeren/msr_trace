/*
 * Normal Program - Secure Implementation
 * Functions with proper bounds checking and security measures
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_BUFFER 64
#define MAX_ITEMS 10

/*
 * Function 1: Safe String Copy with Bounds Checking
 */
int string_processor(const char* input) {
    char buffer[MAX_BUFFER];
    size_t input_len = strlen(input);
    
    // Bounds checking - prevent buffer overflow
    if (input_len >= MAX_BUFFER) {
        return -1;
    }
    
    strcpy(buffer, input);
    printf("[Function 1] Processed: %s\n", buffer);
    return 0;
}

/*
 * Function 2: Safe Array Access with Index Validation
 */
int array_processor(int* data, int count, int index) {
    int local_array[MAX_ITEMS] = {0};
    
    // Validate count bounds
    if (count > MAX_ITEMS || count < 0) {
        return -1;
    }
    
    // Validate index bounds
    if (index < 0 || index >= MAX_ITEMS) {
        return -1;
    }
    
    for (int i = 0; i < count; i++) {
        local_array[i] = data[i] * 2;
    }
    
    // Use validated index to access array
    int result = local_array[index];
    printf("[Function 2] Array[%d] = %d\n", index, result);
    return 0;
}

/*
 * Function 3: Safe Memory Allocation with Validation
 */
int memory_allocator(size_t size, const char* fill_data) {
    // Validate allocation size
    if (size == 0 || size > 1024) {
        return -1;
    }
    
    char* buffer = (char*)malloc(size);
    if (buffer == NULL) {
        return -1;
    }
    
    // Safe fill operation with bounds checking
    size_t fill_len = strlen(fill_data);
    if (fill_len >= size) {
        fill_len = size - 1;
    }
    
    memcpy(buffer, fill_data, fill_len);
    buffer[fill_len] = '\0';
    
    printf("[Function 3] Allocated %zu bytes: %s\n", size, buffer);
    free(buffer);
    return 0;
}

/*
 * Function 4: Safe File Path Handler
 */
int file_path_handler(const char* path, const char* filename) {
    char fullpath[MAX_BUFFER];
    size_t total_len = strlen(path) + strlen(filename) + 2;
    
    // Check combined length
    if (total_len >= MAX_BUFFER) {
        return -1;
    }
    
    snprintf(fullpath, MAX_BUFFER, "%s/%s", path, filename);
    printf("[Function 4] Full path: %s\n", fullpath);
    return 0;
}

/*
 * Function 5: Safe Integer Parser
 */
int parse_integer(const char* str, int* result) {
    // Validate input
    if (str == NULL || strlen(str) == 0) {
        return -1;
    }
    
    // Check for valid integer characters
    for (size_t i = 0; str[i] != '\0'; i++) {
        if (i == 0 && str[i] == '-') continue;
        if (str[i] < '0' || str[i] > '9') {
            return -1;
        }
    }
    
    *result = atoi(str);
    printf("[Function 5] Parsed: %d\n", *result);
    return 0;
}

/*
 * Function 6: Block Processor with Pointer Advancement
 * Processes data in 16-byte blocks, similar to crypto operations
 */
int block_processor(const unsigned char* in, unsigned char* out, size_t len) {
    size_t num_blocks, last_len;
    size_t all_num_blocks;
    
    // Calculate number of complete 16-byte blocks
    num_blocks = len / 16;
    all_num_blocks = num_blocks;
    
    if (num_blocks > 0) {
        size_t max_idx = 0, top = all_num_blocks, processed_bytes = 0;
        
        // Calculate index for processing
        while (top >>= 1)
            max_idx++;
        
        if (max_idx > 64) {
            return -1;
        }
        
        // Process complete blocks
        for (size_t i = 0; i < num_blocks; i++) {
            for (size_t j = 0; j < 16; j++) {
                out[j] = in[j] ^ 0xAA;  // Simple XOR operation
            }
        }
        
        // CRITICAL: Track processed bytes and advance pointers
        processed_bytes = num_blocks * 16;
        in += processed_bytes;
        out += processed_bytes;
        len -= processed_bytes;
    }
    
    // Handle remaining bytes (less than one block)
    last_len = len % 16;
    if (last_len > 0) {
        for (size_t i = 0; i < last_len; i++) {
            out[i] = in[i] ^ 0xAA;
        }
    }
    
    printf("[Function 6] Processed %zu blocks + %zu bytes\n", num_blocks, last_len);
    return 0;
}

int main(int argc, char* argv[]) {
    printf("=== Normal Program - Secure Implementation ===\n\n");
    
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
    
    printf("\nTesting Function 6 (Block Processor):\n");
    unsigned char input[48] = "AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJKKKKLLLL";
    unsigned char output[48] = {0};
    if (block_processor(input, output, 48) != 0) {
        printf("[Function 6] Error: Block processing failed\n");
    }
    
    printf("\n=== All functions executed safely ===\n");
    return 0;
}
