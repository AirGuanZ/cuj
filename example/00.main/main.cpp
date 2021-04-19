#include <iostream>

#include <cuj/cuj.h>

struct Vec3 : cuj::ast::ClassBase<Vec3>
{
    cuj::ast::Value<int> x = new_member<int>(0);
    cuj::ast::Value<int> y = new_member<int>(0);
    cuj::ast::Value<int> z = new_member<int>(0);

    using ClassBase::ClassBase;
};

int main()
{
    cuj::ast::Context context;
    CUJ_SCOPED_CONTEXT(&context);

    context.begin_function("test_func");
    
    $var(int,   x, 10);
    $var(float, y, 2);

    $if(x)
    {
        y = x + y;
    }
    $else
    {
        y = y + y;
    };

    $while(x)
    {
        x = x - 1;
        y = y + x;
    };

    context.end_function();

    cuj::ir::IRBuilder ir_builder;
    context.gen_ir(ir_builder);

    cuj::gen::IRPrinter printer;
    printer.set_indent("  ");
    printer.print(ir_builder.get_prog());

    std::cout << printer.get_result() << std::endl;
}
