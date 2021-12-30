#include <iostream>

#include <cuj.h>

using namespace cuj;

int main()
{
    int n = 0;
    std::cout << "enter n: ";
    std::cin >> n;

    ScopedModule mod;

    Function pow_n = [n](i32 x) mutable
    {
        i64 result = 1;
        i64 base = i64(x);
        while(n)
        {
            if(n & 1)
                result = result * base;
            base = base * base;
            n >>= 1;
        }
        return result;
    };

    MCJIT mcjit;
    mcjit.generate(mod);

    std::cout << "============================= llvm ir =============================" << std::endl;
    std::cout << mcjit.get_llvm_string();
    std::cout << "===================================================================" << std::endl;

    auto pow_n_func = mcjit.get_function(pow_n);
    for(int i = 1; i < 10; ++i)
    {
        std::cout << "pow(" << i << ", "
                  << n << ") = " << pow_n_func(i) << std::endl;
    }
}
