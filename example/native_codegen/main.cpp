#include <iostream>

#include <cuj/cuj.h>

using namespace cuj;

void run()
{
    ScopedContext context;

    unsigned int n = 0;
    std::cout << "Enter n: ";
    std::cin >> n;

    auto pow_n = context.add_function<int64_t>(
        [n]($int x) mutable
    {
        $i64 result = 1;
        $i64 base = x;
        while(n)
        {
            if(n & 1)
                result = result * base;
            base = base * base;
            n >>= 1;
        }
        $return(result);
    });

    gen::LLVMIRGenerator llvm_gen;
    llvm_gen.generate(context.gen_ir());
    std::cout << std::endl << llvm_gen.get_string() << std::endl;

    auto codegen_result = context.gen_native_jit();
    auto pow_n_func = codegen_result.get_symbol(pow_n);

    std::cout << "n = " << n << std::endl;
    for(int i = 0; i <= 9; ++i)
        std::cout << i << " ^ n = " << pow_n_func(i) << std::endl;
}

int main()
{
    try
    {
        run();
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}
