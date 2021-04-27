#include <iostream>

#include <cuj/cuj.h>

using namespace cuj::ast;

void run()
{
    unsigned int n = 0;

    std::cout << "enter n: ";
    std::cin >> n;
    std::cout << std::endl;

    Context ctx;
    CUJ_SCOPED_CONTEXT(&ctx);

    ctx.add_function<int64_t>(
        "pow_n", [n](Value<int> x) mutable
    {
        $var(int64_t, result, 1);
        $var(int64_t, base, x);
        while(n)
        {
            if(n & 1)
                result = result * base;
            base = base * base;
            n >>= 1;
        }
        $return(result);
    });
    
    std::cout << "=========== llvm ir ===========" << std::endl << std::endl;

    cuj::gen::LLVMIRGenerator llvm_gen;
    llvm_gen.generate(ctx.gen_ir());
    std::cout << llvm_gen.get_string() << std::endl;

    std::cout << "=========== result ===========" << std::endl << std::endl;

    auto jit = ctx.gen_native_jit();
    auto pow_n = jit.get_symbol<int64_t(int)>("pow_n");

    if(!pow_n)
    {
        std::cout << "failed to get symbol 'pow_n'" << std::endl;
        return;
    }

    std::cout << "n = " << n << std::endl;
    for(int i = 0; i <= 9; ++i)
        std::cout << i << " ^ n = " << pow_n(i) << std::endl;
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
