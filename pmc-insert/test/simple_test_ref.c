#include <stdio.h>

int addition(int a, int b) {
    return a + b;
}

int subtraction(int a, int b) {
    return a - b;
}

int helper_func(int x) {
    // Call addition from helper_func
    int result = addition(x, 10);
    return result;
}

int main() {
    // Test case 1: Expression statement
    addition(5, 3);
    
    // Test case 2: Assignment statement
    int sum = addition(10, 20);
    
    // Test case 3: From helper function
    int h = helper_func(7);
    
    // Test case 4: Multiple calls in main
    subtraction(100, 50);
    addition(1, 2);
    
    return 0;
}
