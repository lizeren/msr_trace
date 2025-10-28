/*
 * Test case for PMC instrumentation tool
 * Contains the target functions from api_lists.csv
 */
#include <stdio.h>

// Target functions to be wrapped (defined in api_lists.csv)
int addition(int a, int b) {
    return a + b;
}

int subtraction(int a, int b) {
    return a - b;
}

int multiplication(int a, int b) {
    return a * b;
}

// Helper function (not in api_lists.csv - should not be wrapped)
int helper(int x) {
    return x * 2;
}

int main() {
    printf("=== Math Operations Test ===\n");
    int sum;
    // Test case 1: Expression statement
    addition(5, 3);
    
    // Test case 2: Assignment statement
    sum = addition(10, 20);
    
    // // Test case 3: Declaration with initializer
    // int diff = subtraction(50, 30);
    
    // // Test case 4: Return statement
    // int compute() {
    //     return multiplication(6, 7);
    // }
    
    // // Test case 5: If condition
    // if (addition(1, 1)) {
    //     printf("True branch\n");
    // } else {
    //     printf("False branch\n");
    // }
    
    // // Test case 6: While loop
    // int counter = 3;
    // while (subtraction(counter, 0)) {
    //     printf("Counter: %d\n", counter);
    //     counter--;
    // }
    
    // // Test case 7: Nested calls
    // int result = addition(multiplication(2, 3), subtraction(10, 5));
    
    // // Test case 8: Multiple operations
    // int a = addition(1, 2);
    // int b = subtraction(10, 3);
    // int c = multiplication(a, b);
    
    printf("Results: sum=%d, diff=%d, result=%d, c=%d\n", sum, diff, result, c);
    printf("compute()=%d\n", compute());
    
    // Helper should not be wrapped
    int h = helper(5);
    printf("helper=%d\n", h);
    
    return 0;
}

