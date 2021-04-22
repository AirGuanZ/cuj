#include <iostream>

#include <cuj/cuj.h>

using namespace cuj::builtin;

using cuj::ast::Pointer;

int main()
{
    cuj::ast::Context context;
    CUJ_SCOPED_CONTEXT(&context);

    context.begin_function("vec_add");

    $define_arg(float*, A);
    $define_arg(float*, B);
    $define_arg(float*, C);
    $define_arg(int,    N);

    $define_var(int, i, cuda::thread_index_x());

    $define_var(int[4], arr);

    arr[0] = 0;
    arr[1] = 1;
    arr[2] = 2;
    arr[3] = 3;
    
    $if(i < N)
    {
        C[i] = A[i] + B[i];
    };

    context.end_function();

    cuj::ir::IRBuilder ir_builder;
    context.gen_ir(ir_builder);

    //cuj::gen::IRPrinter printer;
    //printer.print(ir_builder.get_prog());
    //std::cout << printer.get_string() << std::endl;

    cuj::gen::LLVMIRGenerator ir_gen;
    ir_gen.generate(ir_builder.get_prog());
    std::cout << ir_gen.get_string() << std::endl;
}
