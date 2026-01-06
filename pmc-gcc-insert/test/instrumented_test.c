#include <stdio.h>
#include <stdlib.h>

int addition(int a, int b) {
    // Add some branches for branch prediction testing
    if (a > b) {
        return a + b;
    } else if (a < b) {
        return b + a;
    } else {
        return a * 2;
    }
}

int subtraction(int a, int b) {
    // Conditional branch
    int result = a - b;
    if (result < 0) {
        result = -result;
    }
    return result;
}

int multiplication(int a, int b) {
    // Multiple branches
    if (a == 0 || b == 0) {
        return 0;
    } else if (a == 1) {
        return b;
    } else if (b == 1) {
        return a;
    } else {
        return a * b;
    }
}

int helper_func(int x) {
    // Call addition from helper_func with branch
    int result;
    if (x > 0) {
        result = addition(x, 10);
    } else {
        result = addition(x, -10);
    }
    return result;
}

int main() {
    int a = 0;
    // Test case 1: Expression statement
    addition(5, 3);
    
    // Test case 2: Assignment statement
    a = subtraction(10, 5);

    // Test case 3: Declaration initializer
    int sum = addition(10, 20);
    
    // Test case 4: From helper function (not a target function)
    int h = helper_func(7);
    
    // Test case 5: Multiple target functions in rows
    subtraction(100, 50);
    addition(1, 2);
    multiplication(3, 4);
    
    // Add more branches
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) {
            addition(i, i + 1);
        } else {
            subtraction(i * 2, i);
        }
    }
    
    // Conditional based on previous results
    if (sum > 20) {
        multiplication(sum, 2);
    } else {
        subtraction(sum, 5);
    }
    
    printf("Hello, World!\n");

    // Return 0 for success (collect_pmc_features.py expects exit code 0)
    addition(1, 2);  // Still call it for measurement
    return 0;
}

