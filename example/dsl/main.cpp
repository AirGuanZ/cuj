#include <iostream>

#include <cuj/cuj.h>

using namespace cuj::ast;

using namespace cuj::builtin::math;

void run()
{
    Context ctx;
    CUJ_SCOPED_CONTEXT(&ctx);

    auto my_dot = ctx.add_function<float>(
        "dot", [](Float2 a, Float2 b)
    {
        $return(dot(a, b));
    });

    ctx.add_function<float>(
        "get_vec_sum", [&]
    {
        $return(my_dot(make_float2(1, 2), make_float2(3, 4)));
    });
    
    cuj::gen::LLVMIRGenerator llvm_gen;
    llvm_gen.generate(ctx.gen_ir());
    std::cout << llvm_gen.get_string() << std::endl;

    auto jit = ctx.gen_native_jit();
    auto entry = jit.get_symbol<float()>("get_vec_sum");
    std::cout << entry() << std::endl;
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
