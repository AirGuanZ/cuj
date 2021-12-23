#include <iostream>

#include <cuj.h>

using namespace cuj;

int main()
{
    int n = 0;
    std::cout << "enter n: ";
    std::cin >> n;
    std::cout << std::endl;

    ScopedModule mod;

    auto pow_n = function("pow_n", [n](i32 x) mutable
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
    });

    MCJIT mcjit;
    mcjit.generate(mod);

    std::cout << mcjit.get_llvm_string() << std::endl;

    auto pow_n_func = mcjit.get_function<int64_t(int32_t)>("pow_n");
    for(int i = 1; i < 10; ++i)
    {
        std::cout << "pow(" << i << ", "
                  << n << ") = " << pow_n_func(i) << std::endl;
    }
}
