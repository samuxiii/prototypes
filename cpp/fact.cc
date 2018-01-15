#include <iostream>

int factorial(int n)
{
    if (n == 1) 
        return 1;
    return factorial(n-1)*n;
}

int main()
{
    std::cout << factorial(5) << std::endl;
}
