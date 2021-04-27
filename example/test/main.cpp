#include <iostream>

#include <cuj/cuj.h>

using namespace cuj::ast;

class Vec3 : public ClassBase<Vec3>
{
public:

    $mem(float, x);
    $mem(float, y);
    $mem(float, z);

    using ClassBase::ClassBase;
};

void run()
{
    Context ctx;
    CUJ_SCOPED_CONTEXT(&ctx);

    auto sum_vec = ctx.add_function<float>(
        "sum_vec", [](Value<Vec3> v)
    {
        $return(v->x + v->y + v->z);
    });

    ctx.add_function<float>(
        "get_vec_sum", [&]
    {
        $var(Vec3, v);
        v->x = 1;
        v->y = 2;
        v->z = 3;
        $return(sum_vec(v));
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
