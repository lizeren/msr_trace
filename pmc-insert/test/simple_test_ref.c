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

    // TODO:
    // if(addition(1,2) == subtraction(5,2)){
    //     // do nothing
    // }

    return addition(1,2);
    
}
