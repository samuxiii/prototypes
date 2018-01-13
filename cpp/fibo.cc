// Example program
#include <iostream>

//i: 0 1 2 3 4 5 6
//p: 0 0 1 1 2 3
//r: 0 1 1 2 3 5 8
int fib(int n) //iterative=> O(n)
{    
    if (n == 0) return 0;
    int i = 0;
    int p = 1;
    int r = 1;
    
    while (i < n-2)
    {
        int tmp = r;
        r += p;
        p = tmp;
        i++;
    }
    
    return r;
}

int fibr (int n) //recursive>= O(2^n)
{
    if (n == 0) return 0;
    if (n == 1) return 1;
    
    return fibr(n-1) + fibr(n-2);
}

int main()
{
    std::cout << fib(0) << ", "<< fib(1) << ", "<< fib(2) << ", "<< fib(3) << ", "<< fib(4) << ", " << fib(5) << ", " << fib(6) << std::endl;
    std::cout << fibr(0) << ", "<< fibr(1) << ", "<< fibr(2) << ", "<< fibr(3) << ", "<< fibr(4) << ", " << fibr(5) << ", " << fib(6) << std::endl;
}
