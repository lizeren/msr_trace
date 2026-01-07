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
int subtraction(int a, int b) {
    // Conditional branch
    int result = a - b;
    if (result < 0) {
        result = -result;
    }
    addition(5, 3); // this shuold not be instrumented
    helper_func(10); // this shuold not be instrumented too
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



int main() {
    int a = 0;

    addition(5, 3);
    
    a = subtraction(10, 5);

    helper_func(10);
    return 0;
}

