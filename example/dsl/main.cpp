#include <iostream>

#include <cuj/cuj.h>

using namespace cuj::ast;

using namespace cuj::builtin::math;

void run()
{
    Context ctx;
    CUJ_SCOPED_CONTEXT(&ctx);
    
    ctx.add_function<float>(
        "test_sqrt", [&]($float x)
    {
        $return(sqrt(x));
    });
    
    cuj::gen::LLVMIRGenerator llvm_gen;
    llvm_gen.generate(ctx.gen_ir());
    std::cout << llvm_gen.get_string() << std::endl;

    auto jit = ctx.gen_native_jit();
    auto entry = jit.get_symbol<float(float)>("test_sqrt");
    std::cout << entry(2) << std::endl;
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
