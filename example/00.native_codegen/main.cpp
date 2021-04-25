#include <iostream>

#include <cuj/cuj.h>

using namespace cuj::ast;

void run()
{
    unsigned int n = 0;
    std::cout << "enter n: ";
    std::cin >> n;

    Context ctx;
    CUJ_SCOPED_CONTEXT(&ctx);

    ctx.add_function<int>(
        "pow_n", [n](Value<int> x) mutable
    {
        $var(int, result, 1);
        while(n)
        {
            if(n & 1)
                result = result * x;
            x = x * x;
            n >>= 1;
        }
        $return(result);
    });

    cuj::gen::LLVMIRGenerator llvm_gen;
    llvm_gen.generate(ctx.gen_ir());
    std::cout << llvm_gen.get_string() << std::endl;

    auto jit = ctx.gen_native_jit();
    auto pow_n = jit.get_symbol<int(int)>("pow_n");

    if(pow_n)
        std::cout << "2 ^ n = " << pow_n(2) << std::endl;
    else
        std::cout << "failed to get symbol 'pow_n'" << std::endl;
    
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
