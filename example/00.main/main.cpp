#include <iostream>

#include <cuj/cuj.h>

using namespace cuj::builtin;
using namespace cuj::ast;

class Vec4 : public ClassBase<Vec4>
{
public:

    $mem(float, x);
    $mem(float, y);
    $mem(float, z);
    $mem(float, w);

    using ClassBase::ClassBase;
};

Value<Vec4> operator+(const Value<Vec4> &a, const Value<Vec4> &b)
{
    $var(Vec4, result);
    result->x = a->x + b->x;
    result->y = a->y + b->y;
    result->z = a->z + b->z;
    result->w = a->w + b->w;
    return result;
}

int main()
{
    Context context;
    CUJ_SCOPED_CONTEXT(&context);

    context.begin_function("vec_add", cuj::ir::Function::Type::Kernel);
    
    $arg(float*, A);
    $arg(float*, B);
    $arg(float*, C);
    $arg(int,    N);

    $var(int, i);

    i = cuda::thread_index_x() + cuda::block_index_x() * cuda::block_dim_x();
    
    $if(i < N)
    {
        C[i] = A[i] + B[i];
    };

    context.end_function();

    cuj::ir::IRBuilder ir_builder;
    context.gen_ir(ir_builder);

    std::cout << "=========== cujitter ir ===========" << std::endl << std::endl;

    cuj::gen::IRPrinter printer;
    printer.print(ir_builder.get_prog());
    std::cout << printer.get_string() << std::endl;

    std::cout << "=========== llvm ir ===========" << std::endl << std::endl;

    cuj::gen::LLVMIRGenerator ir_gen;
    ir_gen.set_target(cuj::gen::LLVMIRGenerator::Target::PTX);
    ir_gen.generate(ir_builder.get_prog());
    std::cout << ir_gen.get_string() << std::endl;

    std::cout << "=========== ptx ===========" << std::endl << std::endl;

    cuj::gen::PTXGenerator ptx_gen;

    ptx_gen.set_target(cuj::gen::PTXGenerator::Target::PTX64);
    ptx_gen.generate(ir_builder.get_prog());
    std::cout << ptx_gen.get_result() << std::endl;
}
