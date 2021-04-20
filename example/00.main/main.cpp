#include <iostream>

#include <cuj/cuj.h>

struct Vec3 : cuj::ast::ClassBase<Vec3>
{
    $member(int, x);
    $member(int, y);
    $member(int, z);
    
    using ClassBase::ClassBase;
};

int main()
{
    cuj::ast::Context context;
    CUJ_SCOPED_CONTEXT(&context);

    context.begin_function("test_func");

    //$var<int>   x = 10;
    //$var<float> y = 2;
    //$var<int>   z = x;

    $define_arg(cuj::ast::Pointer<Vec3>, pv);

    $define_var(Vec3, v);

    v = pv.deref();

    v->x = 1;
    v->y = 2;
    v->z = 3;
    
    //$if(x)
    //{
    //    y = x + y;
    //}
    //$else
    //{
    //    y = v.y + y;
    //};
    //
    //$while(x)
    //{
    //    x = x - 1;
    //    y = y + x;
    //
    //    $if(y - 7.0f)
    //    {
    //        $break;
    //    };
    //};

    context.end_function();

    cuj::ir::IRBuilder ir_builder;
    context.gen_ir(ir_builder);

    cuj::gen::IRPrinter printer;
    printer.print(ir_builder.get_prog());

    std::cout << printer.get_result() << std::endl;
}
